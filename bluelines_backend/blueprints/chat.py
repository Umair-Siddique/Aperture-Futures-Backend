from flask import Blueprint, request, jsonify, current_app
from uuid import uuid4
from datetime import datetime
import logging

chat_bp = Blueprint('chat', __name__)

# Utility function for user authentication
def authenticate_user(access_token):
    supabase = current_app.supabase
    try:
        user_response = supabase.auth.get_user(access_token)
        user = user_response.user
        return user
    except Exception as e:
        logging.exception("Authentication Error")
        return None

# Create a new chat
@chat_bp.route('/new_chat', methods=['POST'])
def create_new_chat():
    data = request.get_json()
    access_token = data.get('access_token')
    title = data.get('title', 'Untitled Chat')

    if not access_token:
        return jsonify({'error': 'Missing access token'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid or expired access token'}), 401

    conversation_id = str(uuid4())
    current_app.supabase.table('conversations').insert({
        'id': conversation_id,
        'user_id': user.id,
        'title': title,
        'created_at': datetime.utcnow().isoformat()
    }).execute()

    return jsonify({
        'message': 'Chat created',
        'conversation_id': conversation_id
    }), 201

# Delete a chat
@chat_bp.route('/delete_chat/<conversation_id>', methods=['DELETE'])
def delete_chat(conversation_id):
    access_token = request.headers.get('Authorization')

    if not access_token:
        return jsonify({'error': 'Missing Authorization header'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid or expired access token'}), 401

    supabase = current_app.supabase

    # Verify ownership of the chat before deletion
    conv_res = supabase.table('conversations').select('*').eq('id', conversation_id).execute()
    conversation = conv_res.data[0] if conv_res.data else None

    if not conversation or conversation['user_id'] != user.id:
        return jsonify({'error': 'Unauthorized or chat not found'}), 403

    # Delete messages first due to foreign key constraints
    supabase.table('messages').delete().eq('conversation_id', conversation_id).execute()
    supabase.table('conversations').delete().eq('id', conversation_id).execute()

    return jsonify({'message': 'Chat deleted successfully'}), 200

# Get all chats for authenticated user
@chat_bp.route('/chats', methods=['POST'])
def get_all_chats():
    data = request.get_json()
    access_token = data.get('access_token')

    if not access_token:
        return jsonify({'error': 'Missing access token'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid or expired access token'}), 401

    res = current_app.supabase.table('conversations').select('*').eq('user_id', user.id).order('created_at', desc=True).execute()
    conversations = res.data if res.data else []

    return jsonify({'conversations': conversations}), 200

# Get all messages inside a specific chat
@chat_bp.route('/chat/<conversation_id>/messages', methods=['POST'])
def get_messages(conversation_id):
    data = request.get_json()
    access_token = data.get('access_token')

    if not access_token:
        return jsonify({'error': 'Missing access token'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid or expired access token'}), 401

    supabase = current_app.supabase

    # Verify ownership of the chat before fetching messages
    conv_res = supabase.table('conversations').select('*').eq('id', conversation_id).execute()
    conversation = conv_res.data[0] if conv_res.data else None

    if not conversation or conversation['user_id'] != user.id:
        return jsonify({'error': 'Unauthorized or chat not found'}), 403

    res = supabase.table('messages').select('*').eq('conversation_id', conversation_id).order('created_at').execute()
    messages = res.data if res.data else []

    return jsonify({'messages': messages}), 200



@chat_bp.route('/update_chat_title/<conversation_id>', methods=['PUT'])
def update_chat_title(conversation_id):
    data = request.get_json()
    access_token = data.get('access_token')
    new_title = data.get('title')

    if not all([access_token, new_title]):
        return jsonify({'error': 'Missing required fields'}), 400

    user = authenticate_user(access_token)
    if not user:
        return jsonify({'error': 'Invalid or expired access token'}), 401

    supabase = current_app.supabase

    # Verify ownership
    conv_res = supabase.table('conversations').select('*').eq('id', conversation_id).execute()
    conversation = conv_res.data[0] if conv_res.data else None

    if not conversation or conversation['user_id'] != user.id:
        return jsonify({'error': 'Unauthorized or chat not found'}), 403

    # Fixed update (removed updated_at)
    supabase.table('conversations').update({
        'title': new_title  # Only update title
    }).eq('id', conversation_id).execute()

    return jsonify({'message': 'Chat title updated successfully'}), 200

