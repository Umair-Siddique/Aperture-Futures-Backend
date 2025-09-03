from flask import Blueprint, request, jsonify, current_app
from .auth import token_required
from datetime import datetime
from math import ceil

conversations_bp = Blueprint('conversations', __name__)
# --- CREATE CONVERSATION ENDPOINT ---
@conversations_bp.route("/create", methods=["POST"])
@token_required
def create_conversation(user):
    """Create a new conversation for the authenticated user, or return existing if title exists."""
    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body is required"}), 400

    title = data.get("title", "New Conversation")

    if not isinstance(title, str) or len(title.strip()) == 0:
        return jsonify({"error": "Title must be a non-empty string"}), 400

    # Trim whitespace
    title = title.strip()

    try:
        # ✅ Check if a conversation with the same title already exists for this user
        existing = (
            current_app.supabase.table("lifelines_conversations")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user.id)
            .eq("title", title)
            .limit(1)
            .execute()
        )

        if existing.data and len(existing.data) > 0:
            conversation = existing.data[0]
            return jsonify({
                "message": "Conversation already exists",
                "conversation": {
                    "id": conversation["id"],
                    "title": conversation["title"],
                    "created_at": conversation["created_at"],
                    "updated_at": conversation["updated_at"]
                }
            }), 200

        # ✅ Otherwise, insert a new conversation
        response = (
            current_app.supabase.table("lifelines_conversations")
            .insert({
                "user_id": user.id,
                "title": title
            })
            .execute()
        )

        if not response.data:
            return jsonify({"error": "Failed to create conversation"}), 500

        conversation = response.data[0]

        return jsonify({
            "message": "Conversation created successfully",
            "conversation": {
                "id": conversation["id"],
                "title": conversation["title"],
                "created_at": conversation["created_at"],
                "updated_at": conversation["updated_at"]
            }
        }), 201

    except Exception as e:
        current_app.logger.error(f"Error creating conversation: {e}")
        return jsonify({"error": "Failed to create conversation"}), 500



# --- UPDATE CONVERSATION TITLE ENDPOINT ---
@conversations_bp.route("/<conversation_id>/update-title", methods=["PUT"])
@token_required
def update_conversation_title(user, conversation_id):
    """Update the title of a specific conversation."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    new_title = data.get("title")
    
    if not isinstance(new_title, str) or len(new_title.strip()) == 0:
        return jsonify({"error": "Title must be a non-empty string"}), 400
    
    # Trim whitespace
    new_title = new_title.strip()
    
    try:
        # Update conversation title (only if user owns it)
        response = current_app.supabase.table('lifelines_conversations').update({
            "title": new_title
        }).eq("id", conversation_id).eq("user_id", user.id).execute()
        
        if not response.data:
            return jsonify({"error": "Conversation not found or access denied"}), 404
        
        conversation = response.data[0]
        
        return jsonify({
            "message": "Conversation title updated successfully",
            "conversation": {
                "id": conversation["id"],
                "title": conversation["title"],
                "updated_at": conversation["updated_at"]
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error updating conversation title: {e}")
        return jsonify({"error": "Failed to update conversation title"}), 500

# --- DELETE CONVERSATION ENDPOINT ---
@conversations_bp.route("/<conversation_id>", methods=["DELETE"])
@token_required
def delete_conversation(user, conversation_id):
    """Delete a specific conversation."""
    try:
        # Delete conversation (only if user owns it)
        response = current_app.supabase.table('lifelines_conversations').delete().eq("id", conversation_id).eq("user_id", user.id).execute()
        
        if not response.data:
            return jsonify({"error": "Conversation not found or access denied"}), 404
        
        return jsonify({
            "message": "Conversation deleted successfully",
            "conversation_id": conversation_id
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error deleting conversation: {e}")
        return jsonify({"error": "Failed to delete conversation"}), 500

# --- LIST CONVERSATIONS ENDPOINT ---
@conversations_bp.route("/list", methods=["GET"])
@token_required
def list_conversations(user):
    """
    Paginated list of conversations for the authenticated user.
    Query params:
      - page: 1-based page number (default 1)
      - limit: items per page (default 10, max 50)
    Response:
      {
        "items": [{"id": "...", "title": "...", "created_at": "...", "updated_at": "..."}, ...],
        "page": 1,
        "limit": 10,
        "total": 42,
        "total_pages": 5,
        "has_next": true,
        "has_prev": false
      }
    """
    try:
        # Parse and validate parameters
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 10))
        except (TypeError, ValueError):
            page = 1
            limit = 10
        
        # Clamp values
        page = max(page, 1)
        limit = max(1, min(limit, 50))  # Limit between 1 and 50
        
        start = (page - 1) * limit
        end = start + limit - 1  # inclusive end for Supabase .range()
        
        # Get conversations for the authenticated user
        response = (
            current_app.supabase
                .table("lifelines_conversations")
                .select("id, title, created_at, updated_at", count="exact")
                .eq("user_id", user.id)
                .order("updated_at", desc=True)
                .order("created_at", desc=True)  # tie-breaker for stable ordering
                .range(start, end)
                .execute()
        )
        
        items = response.data or []
        total = response.count or 0
        total_pages = ceil(total / limit) if total > 0 else 0
        
        return jsonify({
            "items": items,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error listing conversations: {e}")
        return jsonify({"error": "Failed to list conversations"}), 500

# --- GET SINGLE CONVERSATION ENDPOINT ---
@conversations_bp.route("/<conversation_id>", methods=["GET"])
@token_required
def get_conversation(user, conversation_id):
    """Get a specific conversation by ID."""
    try:
        # Get conversation (only if user owns it)
        response = current_app.supabase.table('lifelines_conversations').select(
            "id, title, created_at, updated_at"
        ).eq("id", conversation_id).eq("user_id", user.id).single().execute()
        
        if not response.data:
            return jsonify({"error": "Conversation not found or access denied"}), 404
        
        return jsonify({
            "conversation": response.data
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting conversation: {e}")
        return jsonify({"error": "Failed to get conversation"}), 500

# --- CREATE MESSAGE ENDPOINT ---
@conversations_bp.route("/<conversation_id>/messages", methods=["POST"])
@token_required
def create_message(user, conversation_id):
    """
    Create a new message in a specific conversation.
    Request body:
      {
        "role": "user" | "assistant" | "system",
        "content": "message content"
      }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    role = data.get("role")
    content = data.get("content")
    
    # Validate role
    if not role or role not in ["user", "assistant", "system"]:
        return jsonify({"error": "Role must be 'user', 'assistant', or 'system'"}), 400
    
    # Validate content
    if not isinstance(content, str) or len(content.strip()) == 0:
        return jsonify({"error": "Content must be a non-empty string"}), 400
    
    # Trim whitespace
    content = content.strip()
    
    try:
        # Debug: Log user information
        current_app.logger.info(f"User ID from token: {user.id}")
        current_app.logger.info(f"Conversation ID: {conversation_id}")
        
        # First verify the conversation exists and user owns it
        conversation_response = current_app.supabase.table('lifelines_conversations').select(
            "id, user_id"
        ).eq("id", conversation_id).execute()
        
        current_app.logger.info(f"Conversation response: {conversation_response.data}")
        
        if not conversation_response.data:
            return jsonify({"error": "Conversation not found"}), 404
        
        conversation = conversation_response.data[0]
        current_app.logger.info(f"Conversation user_id: {conversation.get('user_id')}")
        current_app.logger.info(f"Token user_id: {user.id}")
        
        # Check if user owns the conversation
        if conversation.get('user_id') != user.id:
            return jsonify({"error": "Access denied - conversation belongs to different user"}), 403
        
        # Insert new message
        response = current_app.supabase.table('lifelines_messages').insert({
            "conversation_id": conversation_id,
            "user_id": user.id,
            "role": role,
            "content": content
        }).execute()
        
        if not response.data:
            return jsonify({"error": "Failed to create message"}), 500
        
        message = response.data[0]
        
        # Update conversation's updated_at timestamp
        current_app.supabase.table('lifelines_conversations').update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()
        
        return jsonify({
            "message": "Message created successfully",
            "message_data": {
                "id": message["id"],
                "conversation_id": message["conversation_id"],
                "role": message["role"],
                "content": message["content"],
                "created_at": message["created_at"],
                "updated_at": message["updated_at"]
            }
        }), 201
        
    except Exception as e:
        current_app.logger.error(f"Error creating message: {e}")
        return jsonify({"error": "Failed to create message"}), 500

# --- GET MESSAGES FOR CONVERSATION ENDPOINT ---

# --- GET CONVERSATION MESSAGE PAIRS ENDPOINT ---
@conversations_bp.route("/<conversation_id>/message", methods=["GET"])
@token_required
def get_message_pairs(user, conversation_id):
    """
    Get paginated message pairs (user question + AI response) for a specific conversation.
    Query params:
      - page: 1-based page number (default 1)
      - limit: pairs per page (default 10, max 25)
    Response:
      {
        "items": [
          {
            "user_message": {"id": "...", "content": "...", "created_at": "..."},
            "ai_message": {"id": "...", "content": "...", "created_at": "..."}
          },
          ...
        ],
        "page": 1,
        "limit": 10,
        "total": 42,
        "total_pages": 5,
        "has_next": true,
        "has_prev": false
      }
    """
    try:
        # Debug: Log user information
        current_app.logger.info(f"User ID from token: {user.id}")
        current_app.logger.info(f"Conversation ID: {conversation_id}")
        
        # First verify the conversation exists and user owns it
        conversation_response = current_app.supabase.table('lifelines_conversations').select(
            "id, user_id"
        ).eq("id", conversation_id).execute()
        
        current_app.logger.info(f"Conversation response: {conversation_response.data}")
        
        if not conversation_response.data:
            return jsonify({"error": "Conversation not found"}), 404
        
        conversation = conversation_response.data[0]
        current_app.logger.info(f"Conversation user_id: {conversation.get('user_id')}")
        current_app.logger.info(f"Token user_id: {user.id}")
        
        # Check if user owns the conversation
        if conversation.get('user_id') != user.id:
            return jsonify({"error": "Access denied - conversation belongs to different user"}), 403
        
        # Parse and validate parameters
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 10))
        except (TypeError, ValueError):
            page = 1
            limit = 10
        
        # Clamp values
        page = max(page, 1)
        limit = max(1, min(limit, 25))  # Limit between 1 and 25 pairs
        
        # Calculate offset for pairs (each pair = 2 messages)
        pairs_offset = (page - 1) * limit
        messages_offset = pairs_offset * 2
        messages_limit = limit * 2
        
        # Get messages for the conversation (ordered by creation time)
        response = (
            current_app.supabase
                .table("lifelines_messages")
                .select("id, role, content, created_at, updated_at", count="exact")
                .eq("conversation_id", conversation_id)
                .order("created_at", desc=False)  # Oldest first (chronological order)
                .range(messages_offset, messages_offset + messages_limit - 1)
                .execute()
        )
        
        messages = response.data or []
        total_messages = response.count or 0
        total_pairs = total_messages // 2
        total_pages = ceil(total_pairs / limit) if total_pairs > 0 else 0
        
        # Group messages into pairs
        message_pairs = []
        i = 0
        while i < len(messages) - 1:
            user_msg = messages[i]
            ai_msg = messages[i + 1]
            
            # Verify we have a valid user-assistant pair
            if user_msg["role"] == "user" and ai_msg["role"] == "assistant":
                message_pairs.append({
                    "user_message": {
                        "id": user_msg["id"],
                        "content": user_msg["content"],
                        "created_at": user_msg["created_at"]
                    },
                    "ai_message": {
                        "id": ai_msg["id"],
                        "content": ai_msg["content"],
                        "created_at": ai_msg["created_at"]
                    }
                })
                i += 2  # Skip to next pair
            else:
                # If not a valid pair, skip the first message and try again
                i += 1
        
        return jsonify({
            "items": message_pairs,
            "page": page,
            "limit": limit,
            "total": total_pairs,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting message pairs: {e}")
        return jsonify({"error": "Failed to get message pairs"}), 500