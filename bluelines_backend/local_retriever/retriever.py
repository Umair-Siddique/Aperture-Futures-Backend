import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from openai import OpenAI as OpenAIClient 
import anthropic
from langchain_pinecone import PineconeVectorStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.pinecone import PineconeTranslator
from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnablePassthrough
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema.messages import HumanMessage, AIMessage
from config import Config
from pinecone import Pinecone
from typing import Optional, List, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

anthropic_client = anthropic.Anthropic(api_key=Config.anthropic_api_key)

memory = ConversationBufferWindowMemory(k=5, return_messages=True)

pc = Pinecone(api_key=Config.PINECONE_API_KEY)

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=Config.OPENAI_API_KEY,
)

SYSTEM_PROMPT = """
IDENTITY / PERSONA 

• You are **BlueLines LLM**, a seasoned Security‑Council drafting officer.   

• Tone: polite, collegial, formally concise.   

• Always open with “Thank you for your query,” address the user as “you,” and close with “Please let me know if I can assist further.”   
 
• When the user ask ask to draft the resolution or something related to it then only follow the below Template other for simple questions answer in simple words or as short as possible.

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



# Pinecone Vectorstore (corrected to PineconeVectorStore)
vectorstore = PineconeVectorStore(
    embedding=embeddings,
    index_name=Config.UNSC_INDEX_NAME,
    pinecone_api_key=Config.PINECONE_API_KEY,
    text_key="summary"
)

# Define metadata schema using named arguments explicitly
metadata_field_info = [
    AttributeInfo(name="resolution_no", description="UN resolution number, e.g. 2672", type="string"),
    AttributeInfo(name="year", description="Year the resolution was adopted (e.g. 2023)", type="string"),
    AttributeInfo(name="theme", description="Key topics covered, like humanitarian assistance or COVID-19", type="list[string]"),
    AttributeInfo(name="chapter", description="Chapter of the UN Charter referenced, e.g. Chapter VII", type="string"),
    AttributeInfo(name="charter_articles", description="UN Charter articles invoked (e.g. Article 25)", type="list[string]"),
    AttributeInfo(name="entities", description="Specific UN entities or groups involved, like peacekeeping missions (e.g.,UNIFIL, MINUSMA) or committees (e.g., ISIL & Al-Qaida Sanctions Committee).", type="list[string]"),
    AttributeInfo(name="reporting_cycle", description="How often the Secretary-General reports are required (e.g. 60 days)", type="string"),
    AttributeInfo(name="operative_authority", description="Indicates special legal authority, e.g., resolutions Acting under Chapter VII of the Charter (strong, mandatory).", type="string")
    ]

document_content_description = "Full text of the Security Council resolution."

# Initialize LLM for Self-Query Retriever
llm=OpenAI(temperature=0,api_key=Config.OPENAI_API_KEY,)

structured_query_translator = PineconeTranslator()
# Initialize OpenAI client for direct API calls
openai_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)

def gpt4_completion(prompt, model="gpt-4-turbo"):
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3500,
        temperature=0.7,
        stream=True
    )

    complete_response = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            text_chunk = chunk.choices[0].delta.content
            print(text_chunk, end='', flush=True)  # Stream to terminal
            complete_response += text_chunk

    return complete_response

# Self-query retriever setup with structured query translator
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    structured_query_translator=structured_query_translator,  # <-- explicitly added
    verbose=True,
    enable_limit=True
)



def format_document(doc):
        formatted = f"RESOLUTION {doc.metadata['resolution_no']} ({doc.metadata['year']}):\n"
        formatted += f"CHAPTER: {doc.metadata.get('chapter', 'N/A')}\n"
        formatted += f"THEMES: {', '.join(doc.metadata.get('theme', []))}\n"
        formatted += f"ENTITIES: {', '.join(doc.metadata.get('entities', []))}\n"
        formatted += f"AUTHORITY: {doc.metadata.get('operative_authority', 'N/A')}\n"
        formatted += f"URL: {doc.metadata.get('url', 'N/A')}\n"
        formatted += f"SUMMARY: {doc.page_content}\n"
        formatted += "-"*50 + "\n"
        return formatted

def needs_retrieval_llm(query):
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
    response = llm.invoke(prompt).strip().lower()
    return response == "yes"

# Conditional retrieval function
def conditional_retrieval(query):
    TEMPLATE = f"""
You are tasked with transforming the user's query into a structured search by explicitly mapping it to metadata fields where appropriate.

Query: "{query}"

Use the following metadata fields to structure the query. If the query mentions any of these fields, apply filtering using that metadata. Below each field, example values are shown to illustrate correct usage.

- **resolution_no**: UN resolution number (string, exact match).
  - For Example: "2672", "2533"

- **year**: Year the resolution was adopted (string, exact match).
  - For Example: "2023", "2015"

- **theme**: Topics or issues addressed by the resolution (list[string], match any).
  - For Example: ["humanitarian assistance", "COVID-19", "ceasefire"]

- **chapter**: Chapter of the UN Charter referenced (string, exact match).
  - For Example: "Chapter VII", "Chapter VI"

- **charter_articles**: Articles from the UN Charter invoked (list[string], match any).
  - For Example: ["Article 25", "Article 41", "Article 51"]

- **entities**: Countries or organizations mentioned (list[string], match any).
  - For Example: ["Syria", "United Nations", "Security Council", "Russia"]

- **reporting_cycle**: Frequency of Secretary-General reports (string, exact match).
  - For Example: "every 60 days", "biannual", "monthly"

- **operative_authority**: Whether the resolution is binding or non-binding (string, exact match).
  - For Example: "binding", "non-binding"

---

🔄 Transform the query accordingly.

📌 If metadata is not clearly mentioned in the query, do not invent values. Only map based on clearly inferred data.

"""
    if needs_retrieval_llm(query):
        print("\n🔍 Retrieval triggered based on query.")
        retrieved_docs = retriever.invoke(TEMPLATE)
        if retrieved_docs:
            print(f"\n✅ Retrieved {len(retrieved_docs)} documents from Pinecone:")
            for doc in retrieved_docs:
                print(f"- Resolution: {doc.metadata['resolution_no']} ({doc.metadata['year']})")
        else:
            print("\n⚠️ Retrieval triggered but no documents found.")
        return retrieved_docs
    else:
        print("\n🚫 Retrieval not triggered based on query.")
        return []


# Dense Retrieval
def retrieve_documents(
    query: str,
    top_k: int = 5,
    namespace: Optional[str] = None,
    index_name: str = "unsc-index"
) -> List[Tuple[str, dict]]:
    index = pc.Index(name=index_name)
    dense_vector = embeddings.embed_query(query)

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
def run_in_executor(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    return loop.run_in_executor(executor, func, *args, **kwargs)


async def parallel_retrieve(query):
    # Check if retrieval is necessary
    if not needs_retrieval_llm(query):
        print("\n🚫 Retrieval not triggered based on query.")
        return ""

    # Execute both retrievals concurrently
    dense_future = run_in_executor(retrieve_documents, query)
    structured_future = run_in_executor(conditional_retrieval, query)

    # Await results from both
    dense_results, structured_results = await asyncio.gather(dense_future, structured_future)

    # Create set of resolution numbers from structured results to eliminate duplicates
    structured_resolution_numbers = {doc.metadata.get('resolution_no') for doc in structured_results}

    # Filter out dense results that have matching resolution numbers
    filtered_dense_results = [
        (doc_id, metadata) for doc_id, metadata in dense_results
        if metadata.get('resolution_no') not in structured_resolution_numbers
    ]

    # Print metadata separately for debugging
    print("\nDense Retrieval Metadata:")
    for doc_id, metadata in filtered_dense_results:
        print(metadata)

    print("\nStructured Retrieval Metadata:")
    for doc in structured_results:
        print(doc.metadata)

    # Format dense retrieval results (including summary)
    dense_formatted_docs = []
    for doc_id, metadata in filtered_dense_results:
        formatted = f"RESOLUTION {metadata.get('resolution_no', 'N/A')} ({metadata.get('year', 'N/A')}):\n"
        formatted += f"CHAPTER: {metadata.get('chapter', 'N/A')}\n"
        formatted += f"THEMES: {', '.join(metadata.get('theme', []))}\n"
        formatted += f"ENTITIES: {', '.join(metadata.get('entities', []))}\n"
        formatted += f"AUTHORITY: {metadata.get('operative_authority', 'N/A')}\n"
        formatted += f"URL: {metadata.get('url', 'N/A')}\n"
        formatted += f"SUMMARY: {metadata.get('summary', 'N/A')}\n"
        formatted += "-" * 50 + "\n"
        dense_formatted_docs.append(formatted)

    # Format structured retrieval results (including summary)
    structured_formatted_docs = [format_document(doc) for doc in structured_results]

    # Combine results
    combined_context = "\n".join(dense_formatted_docs + structured_formatted_docs)

    return combined_context

# LCEL Runnable chain
chain = RunnablePassthrough.assign(docs=RunnableLambda(lambda inp: conditional_retrieval(inp["query"])))


def build_final_prompt(query, context, memory):
    history = memory.load_memory_variables({})["history"]
    history_text = ""
    for message in history:
        if isinstance(message, HumanMessage):
            history_text += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            history_text += f"AI: {message.content}\n"

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
2. Follow all formatting guidelines in the SYSTEM PROMPT
3. Provide draft text with proper sourcing
4. Include comparative analysis and insertion points
5. Must Provide citation URLs of the referenced resolution documents if available in the given context.
"""
    return final_prompt

# Adjust your chain to utilize parallel retrieval
async def main_execution_loop():
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ")
        if query.strip().lower() == "exit":
            print("Goodbye!")
            break

        context = await parallel_retrieve(query)
        final_prompt = build_final_prompt(query, context, memory)

        print("\nGENERATING RESPONSE:\n")
        response = gpt4_completion(final_prompt)
        memory.save_context({"input": query}, {"output": response})

# To run the async loop
if __name__ == "__main__":
    asyncio.run(main_execution_loop())