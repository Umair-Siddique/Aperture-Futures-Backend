from flask import Blueprint, request, jsonify, current_app
from marshmallow import Schema, fields, ValidationError
import traceback
from functools import wraps


auth_bp = Blueprint('auth', __name__)


# --- Helper Function to Get User Role ---
def get_user_role(user_id: str) -> str or None: # type: ignore
    """Fetches the role of a user from the 'profiles' table."""
    try:
        # Assumes you have a 'profiles' table with 'id' (UUID) and 'role' (text) columns.
        response = current_app.supabase.table('profiles').select('role').eq('id', user_id).single().execute()
        if response.data:
            return response.data.get('role')
        return None
    except Exception as e:
        print(f"Error fetching user role: {e}")
        return None

# --- Authentication Decorators ---
def token_required(f):
    """
    A decorator to protect routes that require a valid user token.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Authorization header is missing or invalid Updated one'}), 401
        
        token = auth_header.split(' ')[1]
        try:
            # Validate the token and get the user
            user_response = current_app.supabase.auth.get_user(token)
            # Make user object available to the route if needed
            kwargs['user'] = user_response.user
        except Exception as e:
            return jsonify({'message': 'Token is invalid or expired Updated one', 'error': str(e)}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """
    A decorator to protect routes that require admin privileges.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Authorization header is missing or invalid'}), 401

        token = auth_header.split(' ')[1]
        try:
            # Step 1: Validate the token
            user_response = current_app.supabase.auth.get_user(token)
            user = user_response.user
            if not user:
                 return jsonify({'message': 'Token is invalid or expired'}), 401

            # Step 2: Check if the user has the 'admin' role
            role = get_user_role(user.id)
            if role != 'admin':
                return jsonify({'message': 'Admin privileges required. Access denied.'}), 403 # Forbidden
            
            # Make user object available to the route
            kwargs['user'] = user

        except Exception as e:
            return jsonify({'message': 'Authentication error', 'error': str(e)}), 401

        return f(*args, **kwargs)
    return decorated_function

# --- User Auth API Endpoints ---

@auth_bp.route('/signup', methods=['POST'])
def signup():
    """Endpoint for creating a new user account."""
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Email and password are required'}), 400

    try:
        user_response = current_app.supabase.auth.sign_up({
            "email": data.get('email'),
            "password": data.get('password'),
        })
        if user_response.user:
            # Remember to set up a trigger in Supabase to create a profile on new user signup.
            return jsonify({'message': 'User created successfully. Please check your email to verify.', 'user_id': user_response.user.id}), 201
        elif user_response.session is None and user_response.user is None:
            return jsonify({'message': 'User already exists or sign-ups are disabled'}), 400
        else:
            return jsonify({'message': 'An unknown error occurred'}), 500
    except Exception as e:
        return jsonify({'message': 'Could not create user', 'error': str(e)}), 500

@auth_bp.route('/signin', methods=['POST'])
def signin():
    """
    Single endpoint for signing in both users and admins.
    Returns role along with the token.
    """
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Email and password are required'}), 400

    try:
        # Step 1: Authenticate
        session_response = current_app.supabase.auth.sign_in_with_password({
            "email": data.get('email'),
            "password": data.get('password')
        })

        if not session_response.session:
            return jsonify({'message': 'Invalid credentials'}), 401

        # Step 2: Get role from profiles table
        user_id = session_response.user.id
        role = get_user_role(user_id)

        if not role:
            role = "user"   # fallback if role not found

        return jsonify({
            'message': f'Successfully signed in as {role} updatedd',
            'access_token': session_response.session.access_token,
            'user_id': user_id,
            'role': role
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'message': 'Authentication failed', 'error': str(e)}), 500
