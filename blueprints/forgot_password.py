from flask import Blueprint, request, jsonify, current_app
from marshmallow import Schema, fields, ValidationError
import traceback
import secrets
import string
from datetime import datetime, timedelta
import uuid
import re
import hashlib
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote_plus


forgot_password_bp = Blueprint('forgot_password', __name__)

# --- Security Configuration ---
SECURITY_CONFIG = {
    'MAX_ATTEMPTS_PER_EMAIL': 3,  # per hour
    'MAX_FAILURES_PER_IP': 10,    # per hour
    'TOKEN_EXPIRY_HOURS': 1,
    'MIN_PASSWORD_LENGTH': 8,
    'REQUIRE_SPECIAL_CHAR': True,
    'REQUIRE_UPPERCASE': True,
    'REQUIRE_LOWERCASE': True,
    'REQUIRE_NUMBER': True,
    'BLOCKED_IPS': set(),  # Add IPs to block if needed
    'SUSPICIOUS_PATTERNS': [
        r'admin@.*',
        r'test@.*',
        r'admin',
        r'password',
        r'123456'
    ]
}

# --- Validation Schemas ---
class ForgotPasswordSchema(Schema):
    email = fields.Email(required=True, error_messages={
        'required': 'Email is required',
        'invalid': 'Please provide a valid email address'
    })

class ResetPasswordSchema(Schema):
    token = fields.Str(required=True, error_messages={
        'required': 'Reset token is required'
    })
    password = fields.Str(required=True, validate=lambda x: validate_password_strength(x), error_messages={
        'required': 'New password is required',
        'validator_failed': 'Password does not meet security requirements'
    })

def utcnow_iso_z() -> str:
    """Return current UTC time as ISO8601 with Z."""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def parse_iso_to_utc(dt_str: str) -> datetime:
    """
    Parse an ISO8601 string to a tz-aware UTC datetime.
    Accepts values with 'Z' or explicit offsets. If naive, assume UTC.
    """
    if not isinstance(dt_str, str):
        raise ValueError("Invalid datetime string")
    # Normalize Z to +00:00 for fromisoformat
    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)



# --- Security Helper Functions ---
def get_client_ip():
    """Get the real client IP address."""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    else:
        return request.remote_addr

def validate_password_strength(password):
    """Validate password strength according to security policy."""
    if len(password) < SECURITY_CONFIG['MIN_PASSWORD_LENGTH']:
        raise ValidationError(f'Password must be at least {SECURITY_CONFIG["MIN_PASSWORD_LENGTH"]} characters long')
    
    if SECURITY_CONFIG['REQUIRE_UPPERCASE'] and not re.search(r'[A-Z]', password):
        raise ValidationError('Password must contain at least one uppercase letter')
    
    if SECURITY_CONFIG['REQUIRE_LOWERCASE'] and not re.search(r'[a-z]', password):
        raise ValidationError('Password must contain at least one lowercase letter')
    
    if SECURITY_CONFIG['REQUIRE_NUMBER'] and not re.search(r'\d', password):
        raise ValidationError('Password must contain at least one number')
    
    if SECURITY_CONFIG['REQUIRE_SPECIAL_CHAR'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValidationError('Password must contain at least one special character')
    
    return True

def is_suspicious_request(email, ip_address):
    """Check if the request is suspicious."""
    # Check blocked IPs
    if ip_address in SECURITY_CONFIG['BLOCKED_IPS']:
        return True, "IP address is blocked"
    
    # Check suspicious email patterns
    for pattern in SECURITY_CONFIG['SUSPICIOUS_PATTERNS']:
        if re.match(pattern, email, re.IGNORECASE):
            return True, "Suspicious email pattern detected"
    
    return False, None

def check_rate_limit(email, ip_address, attempt_type):
    """Check rate limiting using database function."""
    try:
        result = current_app.supabase.rpc('check_rate_limit', {
            'p_email': email,
            'p_ip_address': ip_address,
            'p_attempt_type': attempt_type
        }).execute()
        
        return result.data if result.data else False
    except Exception as e:
        print(f"Rate limit check error: {e}")
        return False

def log_attempt(email, ip_address, user_agent, attempt_type, success):
    """Log attempt to database."""
    try:
        current_app.supabase.rpc('log_reset_attempt', {
            'p_email': email,
            'p_ip_address': ip_address,
            'p_user_agent': user_agent,
            'p_attempt_type': attempt_type,
            'p_success': success
        }).execute()
    except Exception as e:
        print(f"Error logging attempt: {e}")

def generate_secure_token():
    """Generate a cryptographically secure token."""
    return secrets.token_urlsafe(32)

def hash_token(token):
    """Hash token for secure storage."""
    return hashlib.sha256(token.encode()).hexdigest()

def get_user_id_by_email(email: str) -> str | None:
    """
    Resolve a Supabase Auth user id by email (case-insensitive).
    Tries PostgREST on auth.users, then admin.list_users pagination.
    Requires SERVICE ROLE key for direct auth.users access.
    """
    if not email:
        return None
    email_norm = email.strip().lower()

    # --- Fast path A: PostgREST on auth.users (canonical) ---
    try:
        # supabase-py v2 canonical way to hit a non-public schema:
        # use the postgrest client, select from auth.users
        resp = (current_app.supabase
                .postgrest
                .schema('auth')
                .from_('users')
                .select('id,email')
                .eq('email', email_norm)
                .single()
                .execute())
        data = getattr(resp, 'data', None)
        if data and data.get('id'):
            return data['id']
    except Exception as e:
        print(f"[get_user_id_by_email] postgrest.schema('auth').from_('users') failed: {e}")

    # --- Fast path B: Some client versions support schema kw on table() ---
    try:
        # Not all supabase-py builds support schema= on table(); safe to try.
        resp2 = (current_app.supabase
                 .table('users', schema='auth')
                 .select('id,email')
                 .eq('email', email_norm)
                 .single()
                 .execute())
        data2 = getattr(resp2, 'data', None)
        if data2 and data2.get('id'):
            return data2['id']
    except Exception as e:
        print(f"[get_user_id_by_email] table('users', schema='auth') failed: {e}")

    # --- Fallback: paginate admin.list_users (works everywhere w/ service role) ---
    try:
        admin = getattr(current_app.supabase, 'auth', None)
        admin = getattr(admin, 'admin', None)
        if admin is None:
            print("[get_user_id_by_email] admin client missing")
            return None

        page = 1
        per_page = 100
        while True:
            try:
                listing = admin.list_users(page=page, per_page=per_page)
            except TypeError:
                listing = admin.list_users(page, per_page)  # older signatures

            users = getattr(listing, 'users', None) or []
            if not users:
                break

            for u in users:
                u_email = getattr(u, 'email', None)
                if u_email and u_email.strip().lower() == email_norm:
                    return getattr(u, 'id', None)

            if len(users) < per_page:
                break
            page += 1
    except Exception as e:
        print(f"[get_user_id_by_email] admin.list_users failed: {e}")

    return None


# --- API Endpoints with Enhanced Security ---

@forgot_password_bp.route('/request-reset', methods=['POST'])
def request_password_reset():
    """Request a password reset with comprehensive security checks."""
    # local import to avoid top-level dependency if you prefer
    from urllib.parse import quote_plus

    start_time = time.time()
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')

    try:
        # 1) Validate input
        schema = ForgotPasswordSchema()
        data = schema.load(request.get_json() or {})
        email = (data['email'] or '').strip().lower()

        # 2) Security checks
        is_suspicious, _ = is_suspicious_request(email, client_ip)
        if is_suspicious:
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Request blocked for security reasons'}), 403

        # 3) Rate limiting
        if not check_rate_limit(email, client_ip, 'request'):
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Too many requests. Please try again later.'}), 429

        # 4) Generate secure token + hash for storage
        token = generate_secure_token()
        token_hash = hash_token(token)

        # 5) Persist reset record (UTC with Z)
        expires_at_dt = datetime.now(timezone.utc) + timedelta(hours=SECURITY_CONFIG['TOKEN_EXPIRY_HOURS'])
        reset_data = {
            'id': str(uuid.uuid4()),
            'email': email,
            'token': token_hash,
            'expires_at': expires_at_dt.isoformat().replace('+00:00', 'Z'),
            'ip_address': client_ip,
            'user_agent': user_agent,
            'used': False,
            'created_at': utcnow_iso_z()
        }

        try:
            current_app.supabase.table('password_resets').insert(reset_data).execute()
        except Exception as e:
            print(f"[request-reset] Error creating reset record: {e}")
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Service temporarily unavailable. Please try again.'}), 503

        # 6) Build reset URL (prefer configured base if provided)
        # Expect PASSWORD_RESET_URL_BASE like: https://your-frontend/reset-password
        reset_base = (current_app.config.get('PASSWORD_RESET_URL_BASE') or f"{request.host_url.rstrip('/')}/reset-password")
        # URL-encode the token so email clients don't mangle it
        encoded_token = quote_plus(token)
        reset_url = f"{reset_base}?token={encoded_token}"

        # 7) Send email
        try:
            send_reset_email(email, reset_url)
        except Exception as e:
            print(f"[request-reset] Error sending email: {e}")
            # (Optional) best-effort cleanup if email fails:
            # current_app.supabase.table('password_resets').delete().eq('id', reset_data['id']).execute()
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Failed to send reset email. Please try again later.'}), 500

        # 8) Log success
        log_attempt(email, client_ip, user_agent, 'request', True)

        # 9) Add artificial delay to reduce timing side-channels
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        # 10) Always use neutral response (no user enumeration)
        return jsonify({
            'message': 'If an account with this email exists, a password reset link has been sent.',
            'email': email
        }), 200

    except ValidationError as e:
        log_attempt(email if 'email' in locals() else 'unknown', client_ip, user_agent, 'request', False)
        return jsonify({'message': 'Validation error', 'errors': e.messages}), 400
    except Exception:
        traceback.print_exc()
        log_attempt(email if 'email' in locals() else 'unknown', client_ip, user_agent, 'request', False)
        return jsonify({'message': 'An error occurred while processing your request'}), 500



def send_reset_email(to_email, reset_url):
	# Build HTML template
	html = f"""
	<h2>Password Reset</h2>
	<p>Click the link below to reset your password:</p>
	<p><a href="{reset_url}">Reset Password</a></p>
	<p>If you didn’t request this, you can ignore this email.</p>
	"""

	msg = MIMEText(html, "html")
	msg["Subject"] = "Reset your password"

	# Load SMTP config from environment via Flask config
	smtp_host = current_app.config.get("SMTP_HOST", "smtp.protonmail.ch")
	smtp_port = int(current_app.config.get("SMTP_PORT", 587))
	smtp_user = current_app.config.get("SMTP_USER")
	smtp_password = current_app.config.get("SMTP_PASSWORD")
	smtp_from = current_app.config.get("SMTP_FROM", smtp_user)
	smtp_use_tls = bool(current_app.config.get("SMTP_USE_TLS", True))

	msg["From"] = smtp_from
	msg["To"] = to_email

	with smtplib.SMTP(smtp_host, smtp_port) as server:
		if smtp_use_tls:
			server.starttls()
		if smtp_user and smtp_password:
			server.login(smtp_user, smtp_password)
		server.sendmail(msg["From"], [msg["To"]], msg.as_string())

@forgot_password_bp.route('/verify-token', methods=['POST'])
def verify_reset_token():
    """Verify reset token with security checks."""
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    try:
        data = request.get_json() or {}
        raw_token = (data.get('token') or '').strip()
        if not raw_token:
            return jsonify({'message': 'Token is required'}), 400

        # Normalize token (handles URL-encoded values and stray spaces)
        token = unquote_plus(raw_token)
        token_hash = hash_token(token)
        
        # Rate limiting
        if not check_rate_limit('unknown', client_ip, 'verify'):
            log_attempt('unknown', client_ip, user_agent, 'verify', False)
            return jsonify({'message': 'Too many verification attempts. Please try again later.'}), 429
        
        # Get reset record
        try:
            response = current_app.supabase.table('password_resets').select('*').eq('token', token_hash).single().execute()
            reset_record = response.data if response.data else None
        except Exception:
            reset_record = None
        
        if not reset_record:
            log_attempt('unknown', client_ip, user_agent, 'verify', False)
            return jsonify({'message': 'Invalid or expired reset token'}), 400
        
        # Check if token is already used
        if reset_record.get('used', False):
            log_attempt(reset_record['email'], client_ip, user_agent, 'verify', False)
            return jsonify({'message': 'This reset token has already been used'}), 400
        
        # Check if token is expired (tz-aware UTC comparison)
        try:
            exp = parse_iso_to_utc(reset_record['expires_at'])
        except Exception:
            log_attempt(reset_record['email'], client_ip, user_agent, 'verify', False)
            return jsonify({'message': 'Invalid token expiry format'}), 400

        now = datetime.now(timezone.utc)
        if exp < now:
            log_attempt(reset_record['email'], client_ip, user_agent, 'verify', False)
            return jsonify({'message': 'Reset token has expired'}), 400
        
        log_attempt(reset_record['email'], client_ip, user_agent, 'verify', True)
        
        return jsonify({'message': 'Token is valid', 'email': reset_record['email']}), 200
        
    except Exception:
        traceback.print_exc()
        log_attempt('unknown', client_ip, user_agent, 'verify', False)
        return jsonify({'message': 'An error occurred while verifying token'}), 500


@forgot_password_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password with comprehensive security validation."""
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')

    try:
        # ---- 1) Validate input & normalize token ----
        schema = ResetPasswordSchema()
        payload = request.get_json() or {}
        payload['token'] = unquote_plus((payload.get('token') or '').strip())
        data = schema.load(payload)

        token = data['token']
        new_password = data['password']
        token_hash = hash_token(token)

        # ---- 2) Rate limit ----
        if not check_rate_limit('unknown', client_ip, 'reset'):
            log_attempt('unknown', client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Too many reset attempts. Please try again later.'}), 429

        # ---- 3) Lookup reset record ----
        try:
            resp = current_app.supabase.table('password_resets') \
                .select('*').eq('token', token_hash).single().execute()
            reset_record = resp.data if resp and resp.data else None
        except Exception:
            reset_record = None

        if not reset_record:
            log_attempt('unknown', client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Invalid or expired reset token'}), 400

        if reset_record.get('used', False):
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'This reset token has already been used'}), 400

        # ---- 4) Expiry (UTC, tz-aware) ----
        try:
            exp = datetime.fromisoformat(reset_record['expires_at'].replace('Z', '+00:00'))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
        except Exception:
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Invalid token expiry format'}), 400

        if exp < now:
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Reset token has expired'}), 400

        # ---- 5) Ensure Admin client (SERVICE ROLE) is present ----
        admin = getattr(current_app.supabase, 'auth', None)
        admin = getattr(admin, 'admin', None)
        if admin is None:
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Server misconfigured: admin client unavailable.'}), 500

        # quick permission probe
        try:
            try:
                admin.list_users(page=1, per_page=1)
            except TypeError:
                admin.list_users(1, 1)
        except Exception:
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Server is not authorized to change passwords (service role key required).'}), 401

        # ---- 6) Resolve user id by email ----
        user_id = get_user_id_by_email(reset_record['email'])
        if not user_id:
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'User not found'}), 404

        # ---- 7) Update password ----
        try:
            admin.update_user_by_id(user_id, {"password": new_password})
        except Exception as upd_err:
            print(f"Error updating password: {upd_err}")
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({'message': 'Failed to update password. Please try again.'}), 500

        # ---- 8) Mark token as used ----
        try:
            current_app.supabase.table('password_resets').update({
                'used': True,
                'attempts': reset_record.get('attempts', 0) + 1,
                'last_attempt': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
            }).eq('token', token_hash).execute()
        except Exception as upd2_err:
            print(f"Warning: failed to mark token used: {upd2_err}")

        log_attempt(reset_record['email'], client_ip, user_agent, 'reset', True)
        return jsonify({'message': 'Password reset successfully. You can now sign in with your new password.'}), 200

    except ValidationError as e:
        log_attempt('unknown', client_ip, user_agent, 'reset', False)
        return jsonify({'message': 'Validation error', 'errors': e.messages}), 400
    except Exception as e:
        traceback.print_exc()
        log_attempt('unknown', client_ip, user_agent, 'reset', False)
        return jsonify({'message': 'An error occurred while resetting password'}), 500


# --- Security Monitoring Endpoints (Admin Only) ---

@forgot_password_bp.route('/security/stats', methods=['GET'])
def get_security_stats():
    """Get security statistics (admin only)."""
    try:
        # Get recent attempts
        recent_attempts = current_app.supabase.table('password_reset_attempts').select('*').gte('created_at', (datetime.utcnow() - timedelta(hours=24)).isoformat()).execute()
        
        # Get active tokens
        active_tokens = current_app.supabase.table('password_resets').select('*').eq('used', False).gte('expires_at', datetime.utcnow().isoformat()).execute()
        
        stats = {
            'total_attempts_24h': len(recent_attempts.data),
            'successful_attempts_24h': len([a for a in recent_attempts.data if a['success']]),
            'failed_attempts_24h': len([a for a in recent_attempts.data if not a['success']]),
            'active_tokens': len(active_tokens.data),
            'unique_ips_24h': len(set(a['ip_address'] for a in recent_attempts.data)),
            'unique_emails_24h': len(set(a['email'] for a in recent_attempts.data))
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({
            'message': 'Failed to get security stats',
            'error': str(e)
        }), 500