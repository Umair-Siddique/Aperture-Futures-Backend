from flask import Blueprint, request, jsonify, current_app, Response, stream_with_context
import logging
from uuid import uuid4
from datetime import datetime
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from config import Config
from typing import Optional, List, Tuple, Literal
import asyncio
from langchain_openai import OpenAIEmbeddings
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
import json
import re

retriever_bp = Blueprint('bluelines_retriever', __name__)
executor = ThreadPoolExecutor()

# Define state for LangGraph
class GraphState(TypedDict):
    query: str
    query_type: str
    context: str
    app: object

def classify_query_type(query: str, app) -> str:
    """
    Use LLM to classify whether query needs web search (reliefweb), 
    Security Council Report search, or resolution search (Pinecone).
    """
    with app.app_context():
        prompt = f"""
Analyze the following query and determine if it requires:

1. WEB SEARCH (reliefweb, humanitarian reports, WEF documents, current events, news)
   - Examples: "Ukraine Humanitarian situation Report", "Latest WEF report", "Syrian crisis update from reliefweb"
   - Indicators: mentions of humanitarian reports, situation reports, reliefweb, WEF, current news, humanitarian crises

2. SECURITY COUNCIL REPORT SEARCH (Security Council Report website - forecasts, analysis, what's in blue)
   - Examples: "Latest Security Council forecast", "What's in blue for Yemen", "Security Council monthly forecast", "SCR report on Somalia"
   - Indicators: mentions of Security Council Report, monthly forecasts, What's In Blue, SCR publications, Security Council analysis, upcoming Council meetings

3. RESOLUTION SEARCH (UN Security Council resolutions from Pinecone database)
   - Examples: "Draft resolution about climate security", "Find resolutions about North Korea sanctions"
   - Indicators: mentions of resolutions, UNSC resolutions, draft resolutions, sanctions, peacekeeping mandates, resolution text

QUERY TO ANALYZE:
"{query}"

Respond with ONLY one word: "web" or "security_council_report" or "resolution"

RESPONSE:"""
        
        response = app.llm.invoke(prompt).strip().lower()
        
        # Ensure valid response
        if "web" in response:
            return "web"
        elif "security_council_report" in response or "scr" in response:
            return "security_council_report"
        elif "resolution" in response:
            return "resolution"
        else:
            # Default to resolution if unclear
            return "resolution"

def search_tavily(query: str, app) -> str:
    """
    Search using Tavily API with reliefweb focus for humanitarian and current event queries.
    """
    with app.app_context():
        try:
            # Perform Tavily search with provided parameters
            search_params = {
                "query": query,
                "topic": "news",
                "search_depth": "advanced",
                "max_results": 1,
                "include_answer": True,
                "include_raw_content": True,
                "include_images": False,
                "include_domains": ["reliefweb.int"],
            }
            
            response = app.tavily.search(**search_params)
            
            # Format the results
            formatted_results = []
            
            # Add the AI-generated answer if available
            if response.get("answer"):
                formatted_results.append(f"SUMMARY:\n{response['answer']}\n{'-' * 50}\n")
            
            # Add detailed results
            for idx, result in enumerate(response.get("results", []), 1):
                formatted = (
                    f"SOURCE {idx}:\n"
                    f"TITLE: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"CONTENT:\n{result.get('content', 'N/A')}\n"
                )
                
                # Add raw content if available
                if result.get('raw_content'):
                    formatted += f"FULL TEXT:\n{result.get('raw_content')[:2000]}...\n"
                
                formatted += "-" * 50 + "\n"
                formatted_results.append(formatted)
            
            return "\n".join(formatted_results) if formatted_results else "No results found from web search."
            
        except Exception as e:
            logging.error(f"Tavily search error: {e}")
            return f"Error performing web search: {str(e)}"

def search_security_council_report(query: str, app) -> str:
    """
    Search using Tavily API with Security Council Report focus for UNSC analysis, forecasts, and What's In Blue.
    """
    with app.app_context():
        try:
            # Perform Tavily search targeting securitycouncilreport.org
            search_params = {
                "query": query,
                "topic": "general",
                "search_depth": "advanced",
                "max_results": 1,
                "include_answer": True,
                "include_raw_content": True,
                "include_images": False,
                "include_domains": ["securitycouncilreport.org"],
            }
            
            response = app.tavily.search(**search_params)
            
            # Format the results
            formatted_results = []
            
            # Add the AI-generated answer if available
            if response.get("answer"):
                formatted_results.append(f"SUMMARY:\n{response['answer']}\n{'-' * 50}\n")
            
            # Add detailed results
            for idx, result in enumerate(response.get("results", []), 1):
                formatted = (
                    f"SOURCE {idx}:\n"
                    f"TITLE: {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"CONTENT:\n{result.get('content', 'N/A')}\n"
                )
                
                # Add raw content if available
                if result.get('raw_content'):
                    formatted += f"FULL TEXT:\n{result.get('raw_content')[:2000]}...\n"
                
                formatted += "-" * 50 + "\n"
                formatted_results.append(formatted)
            
            return "\n".join(formatted_results) if formatted_results else "No results found from Security Council Report."
            
        except Exception as e:
            logging.error(f"Security Council Report search error: {e}")
            return f"Error performing Security Council Report search: {str(e)}"

def retrieve_documents(
    query: str,
    app,
    top_k: int = 5,
    namespace: Optional[str] = None,
    index_name: str = "unsc-index",
    metadata_filter: Optional[dict] = None
):
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
        if metadata_filter:
            query_kwargs["filter"] = metadata_filter
        response = index.query(**query_kwargs)
        return [(match["id"], match.get("metadata", {})) for match in response.get("matches", [])]


# Helper function to run synchronous retrievers in parallel
def run_in_executor(func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(executor, func, *args)

# LangGraph node functions
def classify_node(state: GraphState) -> GraphState:
    """Node to classify query type"""
    query_type = classify_query_type(state["query"], state["app"])
    state["query_type"] = query_type
    print(f"🔍 Query classified as: {query_type}")
    return state

def web_search_node(state: GraphState) -> GraphState:
    """Node to perform Tavily web search"""
    print("🌐 Performing web search via Tavily...")
    context = search_tavily(state["query"], state["app"])
    state["context"] = context
    return state

def security_council_report_search_node(state: GraphState) -> GraphState:
    """Node to perform Security Council Report search"""
    print("📰 Performing Security Council Report search via Tavily...")
    context = search_security_council_report(state["query"], state["app"])
    state["context"] = context
    return state

def resolution_search_node(state: GraphState) -> GraphState:
    """Node to perform Pinecone resolution search"""
    print("📚 Performing resolution search via Pinecone...")
    
    app = state["app"]
    query = state["query"]
    
    metadata_filter = generate_metadata_filter(query, app)
    
    # Retrieve documents from Pinecone
    dense_results = retrieve_documents(query, app, metadata_filter=metadata_filter)
    structured_results = conditional_retrieval(query, app)
    
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
    state["context"] = combined_context
    return state

def route_query(state: GraphState) -> Literal["web_search", "security_council_report_search", "resolution_search"]:
    """Conditional routing based on query type"""
    if state["query_type"] == "web":
        return "web_search"
    elif state["query_type"] == "security_council_report":
        return "security_council_report_search"
    else:
        return "resolution_search"

# Build LangGraph workflow
def build_retrieval_graph():
    """Build the LangGraph workflow for conditional retrieval"""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("classify", classify_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("security_council_report_search", security_council_report_search_node)
    workflow.add_node("resolution_search", resolution_search_node)
    
    # Set entry point
    workflow.set_entry_point("classify")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "classify",
        route_query,
        {
            "web_search": "web_search",
            "security_council_report_search": "security_council_report_search",
            "resolution_search": "resolution_search"
        }
    )
    
    # Add edges to END
    workflow.add_edge("web_search", END)
    workflow.add_edge("security_council_report_search", END)
    workflow.add_edge("resolution_search", END)
    
    return workflow.compile()

# Create the compiled graph (singleton)
retrieval_graph = build_retrieval_graph()

async def parallel_retrieve(query: str, app):
    """
    Main retrieval function using LangGraph for conditional routing
    between web search (Tavily/ReliefWeb), Security Council Report search (Tavily),
    and resolution search (Pinecone)
    """
    if not needs_retrieval_llm(query, app):
        print("\n🚫 Retrieval not triggered based on query.")
        return ""
    
    # Initialize state
    initial_state = {
        "query": query,
        "query_type": "",
        "context": "",
        "app": app
    }
    
    # Run the graph synchronously (LangGraph doesn't require async for invoke)
    def run_graph():
        result = retrieval_graph.invoke(initial_state)
        return result["context"]
    
    # Execute in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    context = await loop.run_in_executor(executor, run_graph)
    
    return context

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

4. Web search for humanitarian reports, WEF documents, reliefweb content, current events
   - Example: "Ukraine Humanitarian situation Report no 56"
   - Example: "Latest WEF report on global risks"
   - Example: "Syrian crisis update from reliefweb"
   - Example: "Current humanitarian situation in Gaza"

5. Security Council Report search for forecasts, What's In Blue, analysis
   - Example: "Latest Security Council forecast for Yemen"
   - Example: "What's in blue for Somalia"
   - Example: "Security Council Report monthly forecast"
   - Example: "SCR analysis on Libya"

Respond "yes" for ANY of these cases. For all other queries (procedural questions, general UNSC info, 
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


def generate_metadata_filter(query: str, app) -> Optional[dict]:
    """
    Uses the LLM to map a user query to Pinecone metadata filters so we can
    combine dense similarity with structured constraints.
    """
    with app.app_context():
        prompt = f"""
You are mapping Security Council user queries onto metadata filters for Pinecone.
Metadata fields available (see ingestion in upload_resolution pipeline):
- resolution_no (string, exact match)
- year (string or integer)
- theme (list[string])
- chapter (string)
- charter_articles (list[string])
- entities (list[string])
- reporting_cycle (string)
- operative_authority (string)

For the user query below, infer ONLY what is explicit or strongly implied.
Return STRICT JSON with this schema:
{{
  "resolution_no": null|string,
  "year": null|string|int,
  "theme": [],
  "chapter": null|string,
  "charter_articles": [],
  "entities": [],
  "reporting_cycle": null|string,
  "operative_authority": null|string
}}

Use null for unknown scalar fields and [] for unknown list fields.
Do not add commentary.

QUERY:
"{query}"
"""
        try:
            response = app.llm.invoke(prompt)
        except Exception:
            logging.exception("Failed to generate metadata filter")
            return None

        metadata = _safe_parse_json(response)
        if not isinstance(metadata, dict):
            return None

        filter_payload = _build_pinecone_filter(metadata)
        return filter_payload or None


def _safe_parse_json(raw_text: str) -> Optional[dict]:
    """Attempt to parse JSON even if wrapped in extra text/code fences."""
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    try:
        json_str = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_str:
            return json.loads(json_str.group())
    except Exception:
        logging.exception("Unable to parse metadata filter JSON")
    return None


def _build_pinecone_filter(metadata: dict) -> dict:
    """Translate metadata values into Pinecone filter operators."""
    filter_payload = {}

    eq_fields = [
        "resolution_no",
        "year",
        "chapter",
        "reporting_cycle",
        "operative_authority"
    ]
    list_fields = ["theme", "charter_articles", "entities"]

    for field in eq_fields:
        value = metadata.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value != "" and value is not None:
            filter_payload[field] = {"$eq": value}

    for field in list_fields:
        values = metadata.get(field) or []
        if isinstance(values, str):
            values = [values]
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        if cleaned:
            filter_payload[field] = {"$in": cleaned}

    return filter_payload


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

    # --- Determine route/source upfront for status message ---
    def classify_and_route():
        # Directly use existing logic to classify, then route_query function to map to node
        initial_state = {
            "query": query,
            "query_type": "",
            "context": "",
            "app": app
        }
        classified_state = classify_node(initial_state)
        node_route = route_query(classified_state)  # "web_search" | "security_council_report_search" | "resolution_search"
        return node_route

    route = classify_and_route()
    if route == "web_search":
        status_msg = "[STATUS] Searching from Web...\n"
    elif route == "security_council_report_search":
        status_msg = "[STATUS] Searching from Web...\n"
    else:
        status_msg = "[STATUS] Searching from Database...\n"

    async def retrieve_and_build():
        return await parallel_retrieve(query, app)

    def generate_and_store():
        with app.app_context():
            chunks = []
            # Yield status first
            yield status_msg
            context = asyncio.run(retrieve_and_build())
            final_prompt = build_final_prompt_with_history(query, context, history)
            # Then yield LLM streamed output as before
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
