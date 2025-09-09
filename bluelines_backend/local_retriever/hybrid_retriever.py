import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pinecone import Pinecone
from typing import Optional, List, Tuple
from config import Config
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone client
pc = Pinecone(api_key=Config.PINECONE_API_KEY)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=Config.OPENAI_API_KEY
)

def get_index(name: str):
    return pc.Index(name=name)

def retrieve_documents(
    query: str,
    top_k: int = 5,
    namespace: Optional[str] = None,
    index_name: str = "security-council-documents-index"
) -> List[Tuple[str, dict]]:
    index = get_index(index_name)
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

if __name__ == "__main__":
    user_query = "Please give me a document containing resolution 2672."
    retrieved_docs = retrieve_documents(
        query=user_query,
        top_k=5,
        namespace=None,
        index_name="security-council-documents-index"
    )

    # Inspect metadata structure
    print("Retrieved Documents Metadata:")
    for doc_id, meta in retrieved_docs:
        print(f"{doc_id}: {meta}")

    # Ensure the correct key is being used based on the above output
    documents_for_reranking = [
        meta["text"] for _, meta in retrieved_docs
        if "text" in meta and isinstance(meta["text"], str)
    ]

    if not documents_for_reranking:
        raise ValueError("No valid documents found for reranking.")    

    rerank_results = pc.inference.rerank(
        model="bge-reranker-v2-m3",
        query=user_query,
        documents=documents_for_reranking,
        top_n=len(documents_for_reranking),
        return_documents=True
    )

    print("Reranked Documents:")
    for ranked_doc in rerank_results.documents:
        print(ranked_doc)


