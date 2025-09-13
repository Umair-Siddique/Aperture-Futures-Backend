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

    # ---- (C) Fetch last 4-5 conversation messages for context ----
    try:
        conversation_messages = (
            current_app.supabase
            .table('lifelines_messages')
            .select('role, content')
            .eq('conversation_id', conversation_id)
            .order('created_at', desc=True)  
            .limit(10)  # Get last 10 messages (5 exchanges)
            .execute()
        )
        
        # Build conversation context from last 4-5 exchanges
        conversation_context = ""
        if conversation_messages.data:
            # Take the last 8-10 messages (4-5 exchanges)
            recent_messages = conversation_messages.data[-8:] if len(conversation_messages.data) > 8 else conversation_messages.data
            
            print(f"\n=== CONVERSATION CONTEXT FOR CONVERSATION ID: {conversation_id} ===")
            print(f"Total messages found: {len(conversation_messages.data)}")
            print(f"Using recent messages: {len(recent_messages)}")
            print("--- Recent Messages ---")
            
            for i, msg in enumerate(recent_messages):
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    conversation_context += f"User: {content}\n"
                    print(f"[{i+1}] User: {content[:100]}{'...' if len(content) > 100 else ''}")
                elif role == 'assistant':
                    conversation_context += f"Assistant: {content}\n"
                    print(f"[{i+1}] Assistant: {content[:100]}{'...' if len(content) > 100 else ''}")
            
            conversation_context = conversation_context.strip()
            print(f"--- End Messages ---")
            print(f"Conversation context length: {len(conversation_context)} characters")
            print("=" * 60)
        else:
            print(f"\n=== NO CONVERSATION HISTORY FOUND FOR ID: {conversation_id} ===")
            
    except Exception as e:
        # If we can't fetch conversation history, continue without it
        conversation_context = ""
        print(f"Warning: Could not fetch conversation history: {str(e)}")

    # ---- (D) Retrieve context from Pinecone ----
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

    # ---- (E) Build RAG prompt with conversation context ----
    system_prompt = """
You are a Chatbot assistant specialized in the United Nations Security Council (UNSC). 
You have access to the chunked transcript of a Council meeting. 
Your task is to retrieve, summarize, and clarify what was said during the debate with immaculate formatting and luxury readability.

---

### Core Rules
1. Always base your answers strictly on the transcript. 
   - If unsure, reply: “This is not in the transcript provided.”
2. Be neutral and diplomatic. Use factual, UN-style language (“condemned,” “welcomed,” “emphasized,” “reaffirmed”).
3. Never hallucinate numbers, names, or statements.
4. Present all outputs with *clean Markdown formatting, perfect spacing, and professional style.*
5. Maintain *luxury readability* — answers should look like a polished UNSC memo.

---

### UNSC Membership (2025 Hard-Coded)
- *Permanent Members (P5):* China, France, Russian Federation, United Kingdom, United States.
- *Elected Members (E10):*
  - Algeria (2025)
  - Denmark (2026)
  - Greece (2026)
  - Guyana (2025)
  - Pakistan (2026)
  - Panama (2026)
  - Republic of Korea (2025)
  - Sierra Leone (2025)
  - Slovenia (2025)
  - Somalia (2026)

*Classification Rules:*
- These 15 are *Council Members (CM)*.
- Non-members speaking under Rule 37 are *Observers/Invited States*.
- UN officials, experts, NGOs are *Rule 39 Briefers*.

---

### Presidency Rules
- Presidency rotates monthly in English alphabetical order.
- For *September 2025*, the Republic of Korea is President.
- The President:
  - Chairs the meeting procedurally.
  - Also delivers their *national intervention* — always the *last Council Member statement* before Observers/Invited States.
  - In transcripts, this national intervention may not be introduced with “I speak in my national capacity.”
  - When parsing transcripts, assume the *last Council intervention = Presidency’s national statement*.

---

### Response Modes

*1. Standard Q&A Mode*
- When asked “What did [Country] say about [Topic]?”:
  - Provide a *summary (2–4 sentences)* in neutral diplomatic style.
  - Follow with a bulleted list of supporting points.
  - Use clear section headings (###) for readability.

*2. Verbatim Retrieval Mode*
- When asked “What exactly did [Country] say about [Topic]?”:
  - Search the transcript for the country’s intervention.
  - Extract the *verbatim sentences* or passages relevant to the topic.
  - Present them as **blockquotes (>)**.
  - Provide a one-line context summary above the quotes (unless the user requests “only the exact words”).
  - If multiple mentions exist, list them separately under bold subheadings.
  - If the country did not mention the topic, reply: “[Country] did not address [Topic] in this transcript.”

---

### Formatting Standards
- *Headings:* Use ### for main sections.
- *Bold:* Only for subheadings or emphasis (**No spaces inside markers**).
- *Spacing:* One line between sections, no clutter.
- *Bullets:* Use - consistently, keep concise.
- *Quotes:* Always formatted with > and attributed correctly.
- Keep answers tight, professional, and diplomatic.

---

### Example Outputs

*Q: What did France say about Black Sea security?*

### France on Black Sea Security
- Condemned Russian strikes on Black Sea ports.  
- Stressed that attacks worsen global food insecurity.  
- Warned of risks to international shipping confidence.  

---

*Q: What exactly did Denmark say about humanitarian access?*

### Denmark on Humanitarian Access
Denmark underscored the importance of ensuring safe humanitarian operations.

*Verbatim transcript excerpts:*  
> “We underscore the critical importance of safe, sustained, and unhindered humanitarian access.”  
> “Denial of relief and attacks on humanitarian workers are unacceptable and must cease immediately.”
"""

    
    user_prompt = (
        "Using only the following transcript chunks and the provided meeting description, "
        "answer the user's question in Markdown. Consider the conversation context to provide "
        "relevant and coherent responses.\n\n"
        f"### Meeting Description\n{description}\n\n"
        f"### Meeting Members\n{', '.join(members)}\n\n"
        f"### Transcript Chunks\n{combined_context if combined_context else '[no transcript context found]'}\n\n"
    )
    
    # Add conversation context if available
    if conversation_context:
        user_prompt += f"### Recent Conversation Context\n{conversation_context}\n\n"
        print(f"\n=== PROMPT SENT TO LLM ===")
        print(f"System prompt length: {len(system_prompt)} characters")
        print(f"User prompt length: {len(user_prompt)} characters")
        print(f"Conversation context included: {'Yes' if conversation_context else 'No'}")
        print("=" * 60)
    
    user_prompt += f"### Current Question\n{query}\n\n### Answer\n"

    # ---- (F) Save the user's query as a message (role='user') before streaming ----
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

    # ---- (G) Stream LLM response, buffer it, then save as assistant message ----
    def generate_stream_and_persist():
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        assistant_text = ""
        try:
            stream = client.chat.completions.create(
                model="gpt-4o",  # Change this line - gpt-4.1 doesn't exist
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
                    yield delta  # This will stream each character/token as it comes

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
