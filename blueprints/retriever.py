from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
import re
import unicodedata
from openai import OpenAI
from config import Config
from .auth import token_required

retriever_bp = Blueprint('retriever', __name__)

def sanitize_id(name: str) -> str:
    """Sanitize string for Pinecone namespace/vector IDs (ASCII only)."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', normalized)
    return sanitized.lower()

@retriever_bp.route("/query", methods=["POST"])
@token_required
def retrieve_transcript(user):
    query = request.form.get("query")
    title = request.form.get("title")

    if not query or not title:
        return jsonify({"error": "Both 'query' and 'title' are required"}), 400

    # 1) Fetch description + members from Supabase
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

    # 2) Retrieve context from Pinecone
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

    # 3) Build RAG prompt (explicitly ask for Markdown)
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

    # 4) Stream only the assistant's Markdown text (no SSE framing)
    def generate():
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        try:
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=3500,
                temperature=0.7,
                stream=True,  # stream markdown chunks
            )

            for chunk in stream:
                delta = None
                # Primary (OpenAI SDK 1.x) path
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    pass
                # Fallback for any alternative event shape
                if not delta:
                    try:
                        if getattr(chunk, "type", None) in ("token", "response.output_text.delta"):
                            delta = getattr(chunk, "delta", None)
                    except Exception:
                        pass

                if delta:
                    yield delta

        except Exception as e:
            # Emit error inline in the stream so you see it in Postman
            yield f"\n\n> **Streaming error:** `{str(e)}`\n"

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",   # disables Nginx buffering if present
        "Connection": "keep-alive",
    }
    # Important: text/markdown so Postman (and other clients) know it's Markdown
    return Response(stream_with_context(generate()), headers=headers, mimetype="text/markdown; charset=utf-8")