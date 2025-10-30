import os
import faiss
import numpy as np
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from groq import Groq
from dotenv import load_dotenv
import sys
import json
import re
import tempfile
from google.cloud import storage

# Ensure processed_files directory exists
os.makedirs("processed_files_2024", exist_ok=True)

# FAISS Index Setup
dimension = 1024
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))

embeddings_model = OllamaEmbeddings(model='mxbai-embed-large:335m')

# Fetch PDFs from Google Cloud Storage
def fetch_pdfs_from_bucket(bucket_name, folder):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=f"{folder}/")
    return [blob for blob in blobs if blob.name.lower().endswith('.pdf')]

# PDF text extraction
def extract_text_from_blob(blob):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        blob.download_to_filename(tmp_file.name)
        doc = fitz.open(tmp_file.name)
        text = "".join(page.get_text() for page in doc)
        doc.close()
    os.remove(tmp_file.name)
    
    return text.replace("Decides to remain seized of the matter", "")

# Chunking
def chunk_text_recursively(text, chunk_size=2500, chunk_overlap=250):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# Embed and store chunks
def embed_and_store_chunks(chunks, embeddings_model, faiss_index):
    vectors, ids = [], []
    for idx, chunk in enumerate(chunks):
        vector = embeddings_model.embed_query(chunk)
        vectors.append(vector)
        ids.append(idx)
    vectors = np.array(vectors).astype('float32')
    faiss_index.add_with_ids(vectors, np.array(ids))

# Retrieve top chunks
def retrieve_top_k_chunks(query, embeddings_model, faiss_index, chunks, k=15):
    query_embedding = embeddings_model.embed_query(query)
    query_vector = np.array(query_embedding).astype('float32').reshape(1, -1)
    _, indices = faiss_index.search(query_vector, k)
    return [chunks[i] for i in indices[0] if i != -1]

# Generate summary using LLM
def generate_summary(text, client):
    summary_prompt = (
    "Summarize the following Security Council document text clearly and concisely."
    "Do not miss any important point in response"
    "Strictly maintain the original tone and language of the document in your summary."
    "Do not add any information beyond the content of the document. "
    "Do not write any extra Line in the response Start apart from the summarized content"
    "Here is the document:\n"
    f"{text}"
)

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.6,
        max_completion_tokens=1000
    )
    return response.choices[0].message.content

def update_metadata_only(blob, bucket_name, output_dir="processed_files_2012"):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not found.")

    client = Groq(api_key=api_key)
    faiss_index.reset()

    text = extract_text_from_blob(blob)
    chunks = chunk_text_recursively(text)
    embed_and_store_chunks(chunks, embeddings_model, faiss_index)

    retrieved_chunks = retrieve_top_k_chunks("Security Council resolution", embeddings_model, faiss_index, chunks)
    context = "\n".join(retrieved_chunks)

    metadata_response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
        temperature=0.6,
        max_completion_tokens=3000
    )

    metadata = safe_json_extract(metadata_response.choices[0].message.content)
    if metadata is None:
        raise ValueError("Failed metadata extraction.")

    # Load existing JSON file
    output_path = os.path.join(output_dir, f"{os.path.basename(blob.name)}.json")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"{output_path} does not exist (expected an existing file).")

    with open(output_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Overwrite only metadata fields
    for key in [
        "resolution_no", "year", "theme", "chapter",
        "charter_articles", "entities", "reporting_cycle", "operative_authority"
    ]:
        existing_data[key] = metadata.get(key, existing_data.get(key))

    # Save back with summary and url untouched
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated metadata only: {blob.name}")


def safe_json_extract(text):
    try:
        # Clean up and remove code block markers
        clean_text = re.sub(r"```json|```", "", text)
        clean_text = re.sub(r"//.*", "", clean_text)

        # Extract JSON block
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if not match:
            return None

        json_str = match.group()

        # Normalize newlines within strings
        json_str = re.sub(r'(?<!\\)\n', '\\n', json_str)

        # Remove trailing commas
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print("Raw response:\n", text)
        return None


# Main PDF processing
def process_pdf_with_llm(blob, bucket_name):
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not found.")

    client = Groq(api_key=api_key)
    faiss_index.reset()

    text = extract_text_from_blob(blob)
    chunks = chunk_text_recursively(text)
    embed_and_store_chunks(chunks, embeddings_model, faiss_index)

    retrieved_chunks = retrieve_top_k_chunks("Security Council resolution", embeddings_model, faiss_index, chunks)
    context = "\n".join(retrieved_chunks)

    metadata_response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": context}],
        temperature=0.6,
        max_completion_tokens=3000
    )

    metadata = safe_json_extract(metadata_response.choices[0].message.content)
    if metadata is None:
        raise ValueError("Failed metadata extraction.")

    doc = fitz.open(stream=blob.download_as_bytes())
    total_pages = len(doc)
    summaries = [generate_summary("".join(doc[j].get_text() for j in range(i, min(i + 2, total_pages))), client)
                 for i in range(0, total_pages, 2)]

    metadata["summary"] = "\n".join(summaries)
    metadata["url"] = f"https://storage.googleapis.com/{bucket_name}/{blob.name}"

    output_path = os.path.join("processed_files_2024", f"{os.path.basename(blob.name)}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed: {blob.name}")


# Metadata extraction system prompt
system_prompt = """
Extract comprehensive metadata from given Security Council document strictly in the following structured format for efficient embedding and integration into the RAG system. Provide only the metadata fields exactly as specified below without any additional content or explanations:

{
  "resolution_no": "",  // Use case: Identify and retrieve specific resolutions by their unique identifier.
  "year": "",  // Use case: Retrieve documents based on the year they were passed.
  "theme": [""],  // . Use case: Group and retrieve resolutions based on specific thematic areas (eg., Humanitarian, Peacekeeping, Sanctions, Terrorism, WPS/CAAC, Climate, Tech, Political etc).
  "chapter": "",  // Options: "Chapter VI" (peaceful actions; phrases: "Urges", "Recommends") OR "Chapter VII" (binding enforcement; phrases: "Decides", "Demands", "Authorizes use of force"). Use case: Identify enforceability and action type.
  "charter_articles": [""] // Extract and list all UN Charter articles explicitly mentioned or referenced in the resolution (e.g., Article 24, Article 25, Article 27).,
  "entities": [""],  // Use case: Specific UN entities or groups involved, like peacekeeping missions (e.g.,UNIFIL, MINUSMA) or committees (e.g., ISIL & Al-Qaida Sanctions Committee).
  "reporting_cycle": "",  // Use case: Identify Frequency of required reports or updates (e.g., "Secretary-General reports every 90 days")
  "operative_authority": "",  // Use case: Indicates special legal authority (e.g., Write "Chapter VII" if the resolution uses enforcement language like "Decides", "Demands", or "Authorizes use of force"; write "Chapter VI" if the resolution uses mediation or recommendation language like "Urges", "Recommends", or "Calls upon").
}

Context from the UN Charter:
- Chapter VI (peaceful actions): Articles 33-38: Investigative & recommendatory.
- Chapter VII (enforcement actions):
  - Article 39: Threats to peace or acts of aggression.
  - Article 41: Non-military sanctions.
  - Article 42: Military action authorization.
- General Articles:
  - Article 24: UNSC primary responsibility for peace and security.
  - Article 25: UNSC decisions are binding on Member States.
  - Article 27: Veto power by P5 consensus required.
  - Article 4: Admission of new UN members.
  - Article 108: Charter amendments approval.
"""

# Example execution
if __name__ == "__main__":
    bucket_name = "bluelines-rag-bucket"
    folder = "2024"
    target_blob_name = "2024/S_RES_2758_(2024)-EN.pdf"

    blobs = fetch_pdfs_from_bucket(bucket_name, folder)
    
    # Find the specific blob by name
    target_blob = next((blob for blob in blobs if blob.name == target_blob_name), None)
    
    if target_blob:
        try:
            process_pdf_with_llm(target_blob, bucket_name)
        except Exception as e:
            print(f"❌ Failed processing {target_blob.name}: {e}")
    else:
        print(f"❌ Blob '{target_blob_name}' not found in bucket '{bucket_name}'.")

