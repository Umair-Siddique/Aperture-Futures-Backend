from flask import Blueprint, request, jsonify, current_app
from .auth import token_required
from datetime import datetime
from math import ceil

conversations_bp = Blueprint('conversations', __name__)

# --- CREATE CONVERSATION ENDPOINT ---
@conversations_bp.route("/create", methods=["POST"])
@token_required
def create_conversation(user):
    """Create a new conversation for the authenticated user."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body is required"}), 400
    
    title = data.get("title", "New Conversation")
    
    if not isinstance(title, str) or len(title.strip()) == 0:
        return jsonify({"error": "Title must be a non-empty string"}), 400
    
    # Trim whitespace
    title = title.strip()
    
    try:
        # Insert new conversation
        response = current_app.supabase.table('lifelines_conversations').insert({
            "user_id": user.id,
            "title": title
        }).execute()
        
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