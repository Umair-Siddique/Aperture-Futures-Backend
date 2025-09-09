from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
import logging
from uuid import uuid4
from datetime import datetime
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from config import Config
from typing import Optional, List, Tuple
import asyncio
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone

retriever_bp = Blueprint('bluelines_retriever', __name__)
executor = ThreadPoolExecutor()

def retrieve_documents(query: str, app, top_k: int = 5, namespace: Optional[str] = None, index_name: str = "unsc-index"):
    with app.app_context():
        index = app.pc.Index(name=index_name)
        dense_vector = app.embeddings.embed_query(query)
        query_kwargs = {
            "vector": dense_vector,
            "top_k": top_k,
            "include_metadata": True,
            "include_values": False
        }
        if namespace:
            query_kwargs["namespace"] = namespace
        response = index.query(**query_kwargs)
        return [(match["id"], match.get("metadata", {})) for match in response.get("matches", [])]


# Helper function to run synchronous retrievers in parallel
def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, func, *args)

async def parallel_retrieve(query: str, app):
    if not needs_retrieval_llm(query, app):
        print("\n🚫 Retrieval not triggered based on query.")
        return ""

    dense_future = run_in_executor(retrieve_documents, query, app)
    structured_future = run_in_executor(conditional_retrieval, query, app)

    dense_results, structured_results = await asyncio.gather(dense_future, structured_future)

    structured_resolution_numbers = {doc.metadata.get('resolution_no') for doc in structured_results}

    filtered_dense_results = [
        (doc_id, metadata) for doc_id, metadata in dense_results
        if metadata.get('resolution_no') not in structured_resolution_numbers
    ]

    def format_metadata(doc_id, metadata):
        formatted = (
            f"RESOLUTION: {metadata.get('resolution_no', 'N/A')} ({metadata.get('year', 'N/A')})\n"
            f"CHAPTER: {metadata.get('chapter', 'N/A')}\n"
            f"CHARTER ARTICLES: {', '.join(metadata.get('charter_articles', []))}\n"
            f"THEMES: {', '.join(metadata.get('theme', []))}\n"
            f"ENTITIES: {', '.join(metadata.get('entities', []))}\n"
            f"REPORTING CYCLE: {metadata.get('reporting_cycle', 'N/A')}\n"
            f"OPERATIVE AUTHORITY: {metadata.get('operative_authority', 'N/A')}\n"
            f"URL: {metadata.get('url', 'N/A')}\n"
            f"SUMMARY:\n{metadata.get('summary', 'N/A')}\n"
            + "-" * 50 + "\n"
        )
        return formatted

    dense_formatted_docs = [
        format_metadata(doc_id, metadata)
        for doc_id, metadata in filtered_dense_results
    ]

    structured_formatted_docs = [
        format_metadata(
            doc.metadata.get('resolution_no', 'N/A'),
            {
                **doc.metadata,
                "summary": doc.page_content
            }
        )
        for doc in structured_results
    ]

    combined_context = "\n".join(dense_formatted_docs + structured_formatted_docs)
    return combined_context

def needs_retrieval_llm(query: str, app):
    with app.app_context():
        prompt = f"""
As a UN Security Council resolution expert assistant, analyze if the user query requires:
1. Drafting a new resolution on a SPECIFIC TOPIC
   - Example: "Draft resolution about climate security"
   - Example: "Create resolution on Syrian humanitarian access"

2. Searching/retrieving existing resolutions
   - Example: "Find resolutions about North Korea sanctions"
   - Example: "Show resolutions from 2020-2023"

3. Statistical/quantitative requests about resolutions
   - Example: "How many resolutions mention cybersecurity?"
   - Example: "Total resolutions on Darfur since 2010"

Respond "yes" ONLY for these cases. For all other queries (procedural questions, general UNSC info, 
resolution formatting help, or vague requests without specific topics/parameters), respond "no".

QUERY TO ANALYZE:
"{query}"

RESPONSE (ONLY "yes" OR "no"):"""
        response = app.llm.invoke(prompt).strip().lower()
        return response == "yes"


def conditional_retrieval(query: str, app):
    with app.app_context():
        # For now, return empty list since we don't have the complex retriever set up
        # This can be enhanced later when the full dependencies are available
        return []


# LCEL Runnable chain
chain = RunnablePassthrough.assign(docs=RunnableLambda(lambda inp: conditional_retrieval(inp["query"])))


def authenticate_user(access_token, app):
    with app.app_context():
        supabase = app.supabase
        try:
            user_response = supabase.auth.get_user(access_token)
            return user_response.user
        except Exception:
            logging.exception("Authentication Error")
            return None


def build_final_prompt_with_history(query, context, history):
    history_text = ""
    for interaction in history[-5:]:
        sender_label = "User" if interaction["sender"] == "user" else "AI"
        history_text += f"{sender_label}: {interaction['content']}\n"

    final_prompt = f"""
CONVERSATION HISTORY:
{history_text}

USER QUERY:
{query}

CONTEXT FROM SECURITY COUNCIL RESOLUTIONS:
{context}

INSTRUCTIONS:
1. Analyze all provided resolutions, Mainly focus on resolutions which are relevant to given Query.
2. Provide clear and short answers for procedural or simple informational queries.
3. Follow all formatting guidelines in the SYSTEM PROMPT
4. Provide draft text with proper sourcing
5. Include comparative analysis and insertion points
6. **Must ONLY include URLs (citations)** to resolutions that are **highly relevant**, clearly tied to the query, and referenced in your response.

"""
    return final_prompt


def store_message(conversation_id, sender, content, app):
    with app.app_context():
        supabase = app.supabase
        message_id = str(uuid4())
        supabase.table('messages').insert({
            'id': message_id,
            'conversation_id': conversation_id,
            'sender': sender,
            'content': content,
            'created_at': datetime.utcnow().isoformat()
        }).execute()


SYSTEM_PROMPT = """
IDENTITY / PERSONA 

• You are **BlueLines LLM**, a seasoned Security‑Council drafting officer.   

• Tone: polite, collegial, formally concise.   

• Always open with “Thank you for your query,” address the user as “you,” and close with “Please let me know if I can assist further.”   
 
• When user ask a simple question Must give simple and short answer based on user query, If they ask to draft resolutions then answer them in below mentioned Template.

• Mission: transform UNSC precedent into ready‑to‑table products, guide users on insertion points, and refer them to human experts when needed. 

OPENING LINE   

   • Begin: **“Thank you for your query.”**   

   • Add one orienting sentence on the relevant legal frame (e.g., Chapter VII). 

DRAFT LINE   

   • Heading: **“DRAFT TEXT – <SHORT TITLE>”**.   

   • Exactly 10 PPs and 10 OPs, each tagged `(SOURCE_UNSCR_<YEAR>_PP/OP#)`. 

SOURCE SUMMARY LINE   

   • Header: **“SOURCE RATIONALISATION”**.   

   • List *PP/OP #* → one‑sentence reason for inclusion.   

   • End with: “If you’d like deeper background on any source, just let me know!” 

COMPLIANCE LINE   

   • Header **“COMPLIANCE CHECKLIST”**; ≤3 bullets on objectives + thematic best practice. 

COMPARATIVE LINE   

   • Header **“COMPARATIVE ANALYSIS”**; cite 2‑3 key precedents, ≤2 lines each.   

   • Optional “Further Reading” nudge. 

HIGHLIGHT SUGGESTION LINE   

   • Header: **“CANDIDATE INSERTION POINTS”**.   

   • Flag up to five PPs/OPs by number for new thematic language or timing details.   

   • Close with: “Would you like me to highlight these sections for manual editing, or shall I propose wording?” 

INTERACTIVE LINE   

   • Offer up to three concise follow‑up questions (reporting cycle, download, etc.) 

UPDATE LINE (when revising text)   

   • Keep original wording unless explicitly told to change it.   

   • Mark edits with **“// UPDATED”**. 

LIST LINE (for information‑only requests)   

   • Numbered list with one‑sentence blurbs + source tags; end with an offer to draft if desired. 

TONE & STYLE LINE   

   • Friendly‑formal; sentences ≤25 words; strong active verbs (Demands, Decides, Urges). 

TRANSPARENCY LINE   

   • If data is missing or uncertain, state so plainly and suggest next steps rather than hallucinating. 

ESCALATION LINE   

   • If, after reasonable clarification attempts, you cannot meet the user’s request **or** the user expresses dissatisfaction, add this polite referral to the close of your reply:   

     “If you need deeper, bespoke assistance, I can connect you with our human experts at Aperture Futures—just email **bluelines@aperturefurtures.com**.”   

   • Use this only when genuine limitations remain; do **not** over‑recommend. 
"""

@retriever_bp.route('/query', methods=['POST'])
def query_retriever():
    data = request.get_json()
    access_token = data.get('access_token')
    conversation_id = data.get('conversation_id')
    query = data.get('query')
    history = data.get('history', [])
    model_id = data.get('model_id', "claude-sonnet-4-20250514")

    if not all([access_token, conversation_id, query]):
        return jsonify({'error': 'Missing required fields'}), 400

    app = current_app._get_current_object()
    user = authenticate_user(access_token, app)
    if not user:
        return jsonify({'error': 'Invalid or expired access token'}), 401

    store_message(conversation_id, "user", query, app)

    async def retrieve_and_build():
        return await parallel_retrieve(query, app)

    context = asyncio.run(retrieve_and_build())

    final_prompt = build_final_prompt_with_history(query, context, history)

    def generate_and_store():
        with app.app_context():
         chunks = []

         for chunk in generate_llm_response(final_prompt, SYSTEM_PROMPT, app, model_id=model_id):
            if chunk:
                chunks.append(chunk)
                yield chunk  # Stream each piece

        # Now store full message after all chunks are sent
         complete_response = "".join(chunks)
         store_message(conversation_id, "assistant", complete_response, app)
    return Response(stream_with_context(generate_and_store()), mimetype='text/plain')



def generate_llm_response(prompt, system_prompt, app, model_id="claude-sonnet-4-20250514"):
    if model_id == "llama-4-scout-17b-16e-instruct":
        client = app.groq
        response_stream = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5500,
            temperature=0.8,
            stream=True
        )
        for chunk in response_stream:
            delta_content = chunk.choices[0].delta.content
            if delta_content:
                yield delta_content

    elif model_id == "claude-sonnet-4-20250514":
        response_stream = app.anthropic.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            system=system_prompt,
            max_tokens=3500,
            temperature=0.7,
            stream=True
        )
        for event in response_stream:
            if event.type == "content_block_delta" and event.delta.text:
                chunk_to_yield = event.delta.text
                print(f"DEBUG: Yielding chunk: {repr(chunk_to_yield)}")
                yield chunk_to_yield

    else:
        raise ValueError(f"Unsupported model_id provided: {model_id}")
