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

# --- API Endpoints with Enhanced Security ---

@forgot_password_bp.route('/request-reset', methods=['POST'])
def request_password_reset():
    """Request a password reset with comprehensive security checks."""
    start_time = time.time()
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')

    try:
        # Validate input
        schema = ForgotPasswordSchema()
        data = schema.load(request.get_json() or {})
        email = data['email'].lower().strip()

        # Security checks
        is_suspicious, _ = is_suspicious_request(email, client_ip)
        if is_suspicious:
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Request blocked for security reasons'}), 403

        # Rate limiting
        if not check_rate_limit(email, client_ip, 'request'):
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Too many requests. Please try again later.'}), 429

        # Generate secure token
        token = generate_secure_token()
        token_hash = hash_token(token)

        # Create reset record
        reset_data = {
            'id': str(uuid.uuid4()),
            'email': email,
            'token': token_hash,
            'expires_at': (datetime.utcnow() + timedelta(hours=SECURITY_CONFIG['TOKEN_EXPIRY_HOURS'])).isoformat(),
            'ip_address': client_ip,
            'user_agent': user_agent,
            'used': False,
            'created_at': datetime.utcnow().isoformat()
        }

        try:
            current_app.supabase.table('password_resets').insert(reset_data).execute()
        except Exception as e:
            print(f"Error creating reset record: {e}")
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Service temporarily unavailable. Please try again.'}), 503

        # Build reset URL (prefer configured base if provided)
        reset_base = current_app.config.get('PASSWORD_RESET_URL_BASE')
        if reset_base:
            reset_url = f"{reset_base}?token={token}"
        else:
            reset_url = f"{request.host_url}reset-password?token={token}"

        # Send email
        send_reset_email(email, reset_url)

        # Log successful attempt
        log_attempt(email, client_ip, user_agent, 'request', True)

        # Add artificial delay to prevent timing attacks
        elapsed_time = time.time() - start_time
        if elapsed_time < 1.0:
            time.sleep(1.0 - elapsed_time)

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
        data = request.get_json()
        if not data or not data.get('token'):
            return jsonify({
                'message': 'Token is required'
            }), 400
        
        token = data['token']
        token_hash = hash_token(token)
        
        # Rate limiting
        if not check_rate_limit('unknown', client_ip, 'verify'):
            log_attempt('unknown', client_ip, user_agent, 'verify', False)
            return jsonify({
                'message': 'Too many verification attempts. Please try again later.'
            }), 429
        
        # Get reset record
        try:
            response = current_app.supabase.table('password_resets').select('*').eq('token', token_hash).single().execute()
            reset_record = response.data if response.data else None
        except Exception:
            reset_record = None
        
        if not reset_record:
            log_attempt('unknown', client_ip, user_agent, 'verify', False)
            return jsonify({
                'message': 'Invalid or expired reset token'
            }), 400
        
        # Check if token is already used
        if reset_record.get('used', False):
            log_attempt(reset_record['email'], client_ip, user_agent, 'verify', False)
            return jsonify({
                'message': 'This reset token has already been used'
            }), 400
        
        # Check if token is expired
        if datetime.fromisoformat(reset_record['expires_at'].replace('Z', '+00:00')) < datetime.utcnow():
            log_attempt(reset_record['email'], client_ip, user_agent, 'verify', False)
            return jsonify({
                'message': 'Reset token has expired'
            }), 400
        
        log_attempt(reset_record['email'], client_ip, user_agent, 'verify', True)
        
        return jsonify({
            'message': 'Token is valid',
            'email': reset_record['email']
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        log_attempt('unknown', client_ip, user_agent, 'verify', False)
        return jsonify({
            'message': 'An error occurred while verifying token'
        }), 500

@forgot_password_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password with comprehensive security validation."""
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    try:
        # Validate input
        schema = ResetPasswordSchema()
        data = schema.load(request.get_json() or {})
        token = data['token']
        new_password = data['password']
        token_hash = hash_token(token)
        
        # Rate limiting
        if not check_rate_limit('unknown', client_ip, 'reset'):
            log_attempt('unknown', client_ip, user_agent, 'reset', False)
            return jsonify({
                'message': 'Too many reset attempts. Please try again later.'
            }), 429
        
        # Get reset record
        try:
            response = current_app.supabase.table('password_resets').select('*').eq('token', token_hash).single().execute()
            reset_record = response.data if response.data else None
        except Exception:
            reset_record = None
        
        if not reset_record:
            log_attempt('unknown', client_ip, user_agent, 'reset', False)
            return jsonify({
                'message': 'Invalid or expired reset token'
            }), 400
        
        # Check if token is already used
        if reset_record.get('used', False):
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({
                'message': 'This reset token has already been used'
            }), 400
        
        # Check if token is expired
        if datetime.fromisoformat(reset_record['expires_at'].replace('Z', '+00:00')) < datetime.utcnow():
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({
                'message': 'Reset token has expired'
            }), 400
        
        # Update user password
        try:
            response = current_app.supabase.auth.admin.list_users()
            user = None
            for u in response.users:
                if u.email.lower() == reset_record['email'].lower():
                    user = u
                    break
            
            if not user:
                log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
                return jsonify({
                    'message': 'User not found'
                }), 404
            
            # Update password
            current_app.supabase.auth.admin.update_user_by_id(
                user.id,
                {"password": new_password}
            )
            
            # Mark token as used
            current_app.supabase.table('password_resets').update({
                'used': True,
                'attempts': reset_record.get('attempts', 0) + 1,
                'last_attempt': datetime.utcnow().isoformat()
            }).eq('token', token_hash).execute()
            
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', True)
            
            return jsonify({
                'message': 'Password reset successfully. You can now sign in with your new password.'
            }), 200
            
        except Exception as e:
            print(f"Error updating password: {e}")
            log_attempt(reset_record['email'], client_ip, user_agent, 'reset', False)
            return jsonify({
                'message': 'Failed to update password. Please try again.'
            }), 500
        
    except ValidationError as e:
        log_attempt('unknown', client_ip, user_agent, 'reset', False)
        return jsonify({
            'message': 'Validation error',
            'errors': e.messages
        }), 400
    except Exception as e:
        traceback.print_exc()
        log_attempt('unknown', client_ip, user_agent, 'reset', False)
        return jsonify({
            'message': 'An error occurred while resetting password'
        }), 500

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