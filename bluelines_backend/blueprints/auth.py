from flask import Blueprint, request, jsonify, current_app
from marshmallow import Schema, fields, ValidationError
import traceback


auth_bp = Blueprint('auth', __name__)


class SignupSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True, validate=lambda p: len(p) >= 8)

@auth_bp.route('/signup', methods=['POST'])
def signup():
    try:
        data = SignupSchema().load(request.get_json())
        email = data['email']
        password = data['password']

        response = current_app.supabase.auth.sign_up({"email": email, "password": password})

        if response.user:
            return jsonify({
                'message': 'Signup successful',
                'user_id': response.user.id
            }), 201

        if response.error:
            return jsonify({'error': response.error.message}), 400

        return jsonify({'error': 'Unknown signup error'}), 500

    except ValidationError as err:
        return jsonify({'errors': err.messages}), 400
    except Exception as e:
        traceback.print_exc()  # <---- Add this line for debugging
        return jsonify({'error': 'Internal server error'}), 500
    
@auth_bp.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')


    if not email or not password:
        return jsonify({'error': 'Email and password required.'}), 400

    try:
        response = current_app.supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        if response.user:
            return jsonify({
                'message': 'Signin successful',
                'user_id': response.user.id,
                'access_token': response.session.access_token
            }), 200
        else:
            error_message = response.error.message if response.error else 'Signin failed.'
            return jsonify({'error': error_message}), 401

    except Exception as e:
        return jsonify({'error': str(e)}), 500