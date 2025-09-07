from flask import Blueprint, request, jsonify, current_app
from marshmallow import Schema, fields, ValidationError
import traceback
import time
from datetime import datetime, timedelta, timezone
import uuid

forgot_password_bp = Blueprint('forgot_password', __name__)

# --- Security Configuration ---
SECURITY_CONFIG = {
    'MAX_ATTEMPTS_PER_EMAIL': 3,  # per hour
    'MAX_FAILURES_PER_IP': 10,    # per hour
    'MIN_PASSWORD_LENGTH': 8,
    'REQUIRE_SPECIAL_CHAR': True,
    'REQUIRE_UPPERCASE': True,
    'REQUIRE_LOWERCASE': True,
    'REQUIRE_NUMBER': True,
    'BLOCKED_IPS': set(),
    'SUSPICIOUS_PATTERNS': [
        r'admin@.*',
        r'test@.*',
        r'admin',
        r'password',
        r'123456'
    ]
}

VERIFIED_TOKENS = {}  # In production, use Redis or database

# --- Validation Schemas ---
class ForgotPasswordSchema(Schema):
    email = fields.Email(required=True, error_messages={
        'required': 'Email is required',
        'invalid': 'Please provide a valid email address'
    })

class VerifyOTPSchema(Schema):
    email = fields.Email(required=True, error_messages={
        'required': 'Email is required',
        'invalid': 'Please provide a valid email address'
    })
    token = fields.Str(required=True, error_messages={
        'required': 'OTP is required'
    })

class ResetPasswordSchema(Schema):
    email = fields.Email(required=True, error_messages={
        'required': 'Email is required',
        'invalid': 'Please provide a valid email address'
    })
    verification_token = fields.Str(required=True, error_messages={
        'required': 'Verification token is required'
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
    import re
    
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
    import re
    
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
        return True

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

# --- API Endpoints ---

@forgot_password_bp.route('/request-reset', methods=['POST'])
def request_password_reset():
    """Request a password reset using Supabase OTP."""
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

        # 4) Send OTP using Supabase Auth
        try:
            response = current_app.supabase.auth.reset_password_email(email)
            print(f"Supabase reset password response: {response}")
        except Exception as e:
            print(f"Supabase reset password error: {e}")
            log_attempt(email, client_ip, user_agent, 'request', False)
            return jsonify({'message': 'Failed to send reset email. Please try again later.'}), 500

        # 5) Log success
        log_attempt(email, client_ip, user_agent, 'request', True)

        # 6) Add artificial delay to reduce timing side-channels
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        # 7) Always use neutral response (no user enumeration)
        return jsonify({
            'message': 'If an account with this email exists, a password reset OTP has been sent.',
            'email': email
        }), 200

    except ValidationError as e:
        log_attempt(email if 'email' in locals() else 'unknown', client_ip, user_agent, 'request', False)
        return jsonify({'message': 'Validation error', 'errors': e.messages}), 400
    except Exception:
        traceback.print_exc()
        log_attempt(email if 'email' in locals() else 'unknown', client_ip, user_agent, 'request', False)
        return jsonify({'message': 'An error occurred while processing your request'}), 500

@forgot_password_bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP with Supabase and return verification token."""
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    try:
        # 1) Validate input
        schema = VerifyOTPSchema()
        data = schema.load(request.get_json() or {})
        email = (data['email'] or '').strip().lower()
        token = (data['token'] or '').strip()
        
        if not token:
            return jsonify({'message': 'OTP is required'}), 400

        # 2) Basic format validation
        if not token.isdigit() or len(token) != 6:
            return jsonify({'message': 'Invalid OTP format. OTP must be 6 digits.'}), 400

        # 3) Rate limiting
        if not check_rate_limit(email, client_ip, 'verify'):
            return jsonify({'message': 'Too many verification attempts. Please try again later.'}), 429

        # 4) Verify OTP with Supabase
        try:
            # Verify the OTP to get a session
            response = current_app.supabase.auth.verify_otp({
                "email": email,
                "token": token,
                "type": "recovery"
            })
            
            if not response.user:
                return jsonify({'message': 'Invalid or expired OTP'}), 400
            
            # 5) Generate verification token and store it
            verification_token = str(uuid.uuid4())
            VERIFIED_TOKENS[email] = {
                'token': verification_token,
                'expires_at': datetime.now(timezone.utc) + timedelta(minutes=10),  # 10 minutes expiry
                'user_id': response.user.id
            }
                
            return jsonify({
                'message': 'OTP verified successfully. You can now reset your password.',
                'user_id': response.user.id,
                'verified': True,
                'verification_token': verification_token
            }), 200
                
        except Exception as e:
            print(f"Supabase OTP verification error: {e}")
            return jsonify({'message': 'Invalid or expired OTP'}), 400
        
    except ValidationError as e:
        return jsonify({'message': 'Validation error', 'errors': e.messages}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({'message': 'An error occurred while verifying OTP'}), 500

@forgot_password_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password using verification token from OTP verification."""
    client_ip = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')

    try:
        # 1) Validate input
        schema = ResetPasswordSchema()
        data = schema.load(request.get_json() or {})
        email = (data['email'] or '').strip().lower()
        verification_token = data['verification_token']
        new_password = data['password']

        # 2) Check if user has a valid verification token
        if email not in VERIFIED_TOKENS:
            return jsonify({'message': 'Please verify your OTP first'}), 400
        
        verification_data = VERIFIED_TOKENS[email]
        
        # 3) Verify the provided verification token matches
        if verification_data['token'] != verification_token:
            return jsonify({'message': 'Invalid verification token'}), 400
        
        # 4) Check if verification token is expired
        if datetime.now(timezone.utc) > verification_data['expires_at']:
            del VERIFIED_TOKENS[email]  # Clean up expired token
            return jsonify({'message': 'Verification token expired. Please verify OTP again.'}), 400

        # 5) Rate limiting
        if not check_rate_limit(email, client_ip, 'reset'):
            return jsonify({'message': 'Too many reset attempts. Please try again later.'}), 429

        # 6) Update password using admin API
        try:
            # Get admin client
            admin = getattr(current_app.supabase_admin, 'auth', None)
            admin = getattr(admin, 'admin', None)
            
            if admin is None:
                return jsonify({'message': 'Server misconfigured'}), 500
            
            # Update password using the stored user_id
            update_response = admin.update_user_by_id(verification_data['user_id'], {
                "password": new_password
            })
            
            print(f"Password update response: {update_response}")
            
            if update_response.user:
                # Clean up verification token after successful password reset
                del VERIFIED_TOKENS[email]
                return jsonify({'message': 'Password reset successfully. You can now sign in with your new password.'}), 200
            else:
                return jsonify({'message': 'Failed to update password. Please try again.'}), 500
            
        except Exception as e:
            print(f"Supabase password reset error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'message': 'Failed to reset password. Please try again.'}), 500

    except ValidationError as e:
        return jsonify({'message': 'Validation error', 'errors': e.messages}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({'message': 'An error occurred while resetting password'}), 500

# --- Security Monitoring Endpoints ---

@forgot_password_bp.route('/security/stats', methods=['GET'])
def get_security_stats():
    """Get security statistics (admin only)."""
    try:
        # Get recent attempts
        recent_attempts = current_app.supabase.table('password_reset_attempts').select('*').gte('created_at', (datetime.utcnow() - timedelta(hours=24)).isoformat()).execute()
        
        stats = {
            'total_attempts_24h': len(recent_attempts.data),
            'successful_attempts_24h': len([a for a in recent_attempts.data if a['success']]),
            'failed_attempts_24h': len([a for a in recent_attempts.data if not a['success']]),
            'unique_ips_24h': len(set(a['ip_address'] for a in recent_attempts.data)),
            'unique_emails_24h': len(set(a['email'] for a in recent_attempts.data))
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({
            'message': 'Failed to get security stats',
            'error': str(e)
        }), 500