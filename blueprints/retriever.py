from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
import re
import unicodedata
from openai import OpenAI
from config import Config
from .auth import token_required
from datetime import datetime

retriever_bp = Blueprint('retriever', __name__)

def sanitize_id(name: str) -> str:
    """Sanitize string for Pinecone namespace/vector IDs (ASCII only)."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', normalized)
    return sanitized.lower()

@retriever_bp.route("/query", methods=["POST"])
@token_required
def retrieve_transcript(user):
    # Expect: query, title (for Pinecone namespace), conversation_id (to log messages)
    query = request.form.get("query")
    title = request.form.get("title")
    conversation_id = request.form.get("conversation_id")

    if not query or not title or not conversation_id:
        return jsonify({"error": "Fields 'query', 'title', and 'conversation_id' are required"}), 400

    # ---- (A) Verify conversation ownership ----
    try:
        conv = (
            current_app.supabase
            .table('lifelines_conversations')
            .select('id, user_id')
            .eq('id', conversation_id)
            .single()
            .execute()
        )
        if not conv.data or conv.data.get('user_id') != user.id:
            return jsonify({"error": "Conversation not found or access denied"}), 403
    except Exception as e:
        return jsonify({"error": f"Supabase error (conversation check): {str(e)}"}), 500

    # ---- (B) Fetch description + members from Supabase (for retrieval prompt) ----
    try:
        record = (
            current_app.supabase
            .table('audio_files')
            .select('description, members')
            .eq('title', title)
            .execute()
        )
    except Exception as e:
        return jsonify({"error": f"Supabase error: {str(e)}"}), 500

    if not record.data:
        return jsonify({"error": f"No record found for title '{title}'"}), 404

    description = record.data[0].get('description', '')
    raw_members = record.data[0].get('members', [])
    if isinstance(raw_members, str):
        members = raw_members.strip("{}").split(",") if raw_members else []
    else:
        members = raw_members or []

    # ---- (C) Retrieve context from Pinecone ----
    try:
        query_embedding = current_app.embeddings.embed_query(query)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500

    safe_namespace = sanitize_id(title)

    try:
        result = current_app.pinecone_index.query(
            vector=query_embedding,
            top_k=7,
            namespace=safe_namespace,
            include_metadata=True
        )
    except Exception as e:
        return jsonify({"error": f"Pinecone query failed: {str(e)}"}), 500

    contexts = [
        m.get('metadata', {}).get('text', '')
        for m in result.get('matches', [])
        if m.get('metadata')
    ]
    combined_context = "\n".join(c for c in contexts if c).strip()

    # ---- (D) Build RAG prompt ----
    system_prompt = (
        "You are a helpful assistant for meeting Q&A. "
        "Always respond in **valid GitHub-Flavored Markdown**. "
        "Use headings, bullet points, numbered lists, and code blocks if useful."
    )
    user_prompt = (
        "Using only the following transcript chunks and the provided meeting description, "
        "answer the user's question in Markdown.\n\n"
        f"### Meeting Description\n{description}\n\n"
        f"### Meeting Members\n{', '.join(members)}\n\n"
        f"### Transcript Chunks\n{combined_context if combined_context else '[no transcript context found]'}\n\n"
        f"### Question\n{query}\n\n"
        "### Answer\n"
    )

    # ---- (E) Save the user's query as a message (role='user') before streaming ----
    try:
        current_app.supabase.table('lifelines_messages').insert({
            "conversation_id": conversation_id,
            "user_id": user.id,
            "role": "user",
            "content": query
        }).execute()

        # bump parent conversation updated_at
        current_app.supabase.table('lifelines_conversations').update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()
    except Exception as e:
        return jsonify({"error": f"Failed to persist user query: {str(e)}"}), 500

    # ---- (F) Stream LLM response, buffer it, then save as assistant message ----
    def generate_stream_and_persist():
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        assistant_text = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=3500,
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                delta = None
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    pass
                if not delta:
                    try:
                        if getattr(chunk, "type", None) in ("token", "response.output_text.delta"):
                            delta = getattr(chunk, "delta", None)
                    except Exception:
                        pass

                if delta:
                    assistant_text += delta
                    yield delta

        except Exception as e:
            err_md = f"\n\n> **Streaming error:** `{str(e)}`\n"
            assistant_text_local = assistant_text + err_md
            # Still try to persist what we have, including error message
            try:
                current_app.supabase.table('lifelines_messages').insert({
                    "conversation_id": conversation_id,
                    "user_id": user.id,
                    "role": "assistant",
                    "content": assistant_text_local
                }).execute()
                current_app.supabase.table('lifelines_conversations').update({
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", conversation_id).execute()
            except Exception:
                pass
            yield err_md
            return

        # Persist the full assistant message and bump updated_at
        try:
            current_app.supabase.table('lifelines_messages').insert({
                "conversation_id": conversation_id,
                "user_id": user.id,
                "role": "assistant",
                "content": assistant_text
            }).execute()
            current_app.supabase.table('lifelines_conversations').update({
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", conversation_id).execute()
        except Exception as e:
            # Emit a footer note so the client knows persistence failed
            yield f"\n\n> **Note:** Failed to save assistant message: `{str(e)}`\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return Response(
        stream_with_context(generate_stream_and_persist()),
        headers=headers,
        mimetype="text/markdown; charset=utf-8"
    )
