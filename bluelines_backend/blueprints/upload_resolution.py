from flask import Blueprint, request, jsonify, current_app
import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import json
import re
import logging
from werkzeug.utils import secure_filename
from openai import OpenAI
from uuid import uuid4

load_dotenv()

upload_resolution_bp = Blueprint('upload_resolution', __name__)

# Configuration
ALLOWED_EXTENSIONS = {'pdf'}
EMBEDDING_MODEL = 'text-embedding-3-large'
TEMP_NAMESPACE_PREFIX = 'temp_upload_'
NEW_RESOLUTIONS_NAMESPACE = 'new_resolutions'

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
        # Remove common footer text
        return text.replace("Decides to remain seized of the matter", "")
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise

def chunk_text_recursively(text, chunk_size=2500, chunk_overlap=250):
    """Split text into chunks using RecursiveCharacterTextSplitter"""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def embed_and_store_chunks_in_pinecone(chunks, openai_client, pinecone_index, temp_namespace):
    """Embed chunks and store in Pinecone temporary namespace"""
    vectors_to_upsert = []
    
    # Process chunks in batches for efficiency
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        response = openai_client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        
        for idx, embedding_obj in enumerate(response.data):
            chunk_idx = i + idx
            vectors_to_upsert.append({
                'id': f"chunk_{chunk_idx}",
                'values': embedding_obj.embedding,
                'metadata': {
                    'chunk_text': chunks[chunk_idx],
                    'chunk_index': chunk_idx
                }
            })
    
    # Upsert all vectors to temporary namespace
    pinecone_index.upsert(vectors=vectors_to_upsert, namespace=temp_namespace)
    
    return len(vectors_to_upsert)

def retrieve_top_k_chunks_from_pinecone(query, openai_client, pinecone_index, temp_namespace, k=15):
    """Retrieve top-k most relevant chunks from Pinecone temporary namespace"""
    # Generate query embedding
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    query_embedding = response.data[0].embedding
    
    # Query Pinecone temporary namespace
    query_response = pinecone_index.query(
        vector=query_embedding,
        top_k=k,
        namespace=temp_namespace,
        include_metadata=True
    )
    
    # Extract chunk texts from results
    retrieved_chunks = []
    for match in query_response.matches:
        chunk_text = match.metadata.get('chunk_text', '')
        if chunk_text:
            retrieved_chunks.append(chunk_text)
    
    return retrieved_chunks

def generate_summary(text, client):
    """Generate summary using Groq LLM"""
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

def safe_json_extract(text):
    """Safely extract JSON from LLM response"""
    try:
        # Clean up and remove code block markers
        clean_text = re.sub(r"```json|```|```", "", text)
        clean_text = re.sub(r"//.*", "", clean_text)

        # Find all JSON objects (in case LLM returns multiple)
        json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_text)
        
        if not json_objects:
            return None
        
        # Try each JSON object until one parses successfully
        for json_str in json_objects:
            try:
                # Normalize newlines within strings
                json_str = re.sub(r'(?<!\\)\n', '\\n', json_str)
                
                # Remove trailing commas
                json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
                
                parsed = json.loads(json_str)
                
                # Validate it has the expected structure
                if isinstance(parsed, dict) and "resolution_no" in parsed:
                    if len(json_objects) > 1:
                        logging.warning(f"Multiple JSON objects found, using the first valid one (Resolution {parsed.get('resolution_no')})")
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # If no valid JSON found, try the old method as fallback
        match = re.search(r'\{.*?\}', clean_text, re.DOTALL)
        if match:
            json_str = match.group()
            json_str = re.sub(r'(?<!\\)\n', '\\n', json_str)
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
            return json.loads(json_str)
            
        return None
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {e}")
        logging.error(f"Raw response:\n{text}")
        return None

def store_in_pinecone_permanent(metadata, original_filename, openai_client, pinecone_index):
    """Store the processed document embedding in Pinecone permanent namespace"""
    try:
        # Generate embedding for the summary
        content = metadata.get('summary', '')
        if not content:
            raise ValueError("Summary is empty, cannot generate embedding")
        
        response = openai_client.embeddings.create(
            input=[content],
            model=EMBEDDING_MODEL
        )
        
        embedding = response.data[0].embedding
        
        # Prepare metadata for Pinecone
        pinecone_metadata = {
            'resolution_no': metadata.get('resolution_no'),
            'year': metadata.get('year'),
            'theme': metadata.get('theme', []),
            'chapter': metadata.get('chapter'),
            'charter_articles': metadata.get('charter_articles', []),
            'entities': metadata.get('entities', []),
            'reporting_cycle': metadata.get('reporting_cycle'),
            'operative_authority': metadata.get('operative_authority'),
            'summary': content,
            'url': metadata.get('url', '')
        }
        
        # Create unique ID using original filename (without UUID prefix)
        resolution_no = metadata.get('resolution_no', 'unknown')
        year = metadata.get('year', 'unknown')
        basename = os.path.splitext(original_filename)[0]
        vector_id = f"{basename}_{resolution_no}_{year}"
        
        # Upsert to Pinecone in new_resolutions namespace
        pinecone_index.upsert(vectors=[{
            'id': vector_id,
            'values': embedding,
            'metadata': pinecone_metadata
        }], namespace=NEW_RESOLUTIONS_NAMESPACE)
        
        logging.info(f"Successfully stored in Pinecone namespace '{NEW_RESOLUTIONS_NAMESPACE}' with ID: {vector_id}")
        return vector_id
        
    except Exception as e:
        logging.error(f"Error storing in Pinecone: {e}")
        raise

def cleanup_temp_namespace(pinecone_index, temp_namespace):
    """Delete all vectors from temporary namespace"""
    try:
        # Delete all vectors in the temporary namespace by deleting the namespace
        pinecone_index.delete(delete_all=True, namespace=temp_namespace)
        logging.info(f"Successfully cleaned up temporary namespace: {temp_namespace}")
    except Exception as e:
        logging.error(f"Error cleaning up temporary namespace {temp_namespace}: {e}")
        # Don't raise - cleanup failure shouldn't fail the request

def process_pdf_with_openai(pdf_file, filename, app):
    """Main PDF processing function using OpenAI embeddings and Pinecone"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    index_name = os.getenv('UNSC_INDEX_NAME')
    if not index_name:
        raise ValueError("UNSC_INDEX_NAME not found in environment variables")

    openai_client = OpenAI(api_key=openai_api_key)
    
    # Get services from app context (Groq and Pinecone)
    with app.app_context():
        if not hasattr(app, 'groq'):
            raise ValueError("Groq client not initialized. Make sure init_groq() is called.")
        groq_client = app.groq
        pinecone_index = app.pc.Index(index_name)
    
    # Extract original filename from unique_filename (remove UUID prefix if present)
    # Format: {uuid}_{original_filename} -> extract original_filename
    original_filename = filename
    if '_' in filename:
        # Check if it starts with a UUID-like pattern (32 hex chars)
        parts = filename.split('_', 1)
        if len(parts) == 2 and len(parts[0]) == 32 and all(c in '0123456789abcdef' for c in parts[0].lower()):
            original_filename = parts[1]
    
    # Create unique temporary namespace for this upload
    temp_namespace = f"{TEMP_NAMESPACE_PREFIX}{uuid4().hex}"
    logging.info(f"Using temporary namespace: {temp_namespace}")
    
    try:
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_file)
        
        # Chunk the text
        chunks = chunk_text_recursively(text)
        logging.info(f"Created {len(chunks)} chunks from PDF")
        
        # Embed and store chunks in Pinecone temporary namespace
        num_stored = embed_and_store_chunks_in_pinecone(
            chunks, openai_client, pinecone_index, temp_namespace
        )
        logging.info(f"Stored {num_stored} chunks in temporary namespace")

        # Retrieve top chunks for metadata extraction from Pinecone
        retrieved_chunks = retrieve_top_k_chunks_from_pinecone(
            "Security Council resolution", 
            openai_client, 
            pinecone_index,
            temp_namespace
        )
        context = "\n".join(retrieved_chunks)
        logging.info(f"Retrieved {len(retrieved_chunks)} chunks for metadata extraction")

        # Extract metadata using LLM
        metadata_response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": METADATA_SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ],
            temperature=0.6,
            max_completion_tokens=3000
        )

        metadata = safe_json_extract(metadata_response.choices[0].message.content)
        if metadata is None:
            raise ValueError("Failed to extract metadata from the document")
        
        # Clean up resolution_no: remove year in brackets if present
        if "resolution_no" in metadata and metadata["resolution_no"]:
            # Remove patterns like " (1947)", " (2023)", etc.
            metadata["resolution_no"] = re.sub(r'\s*\(\d{4}\)', '', metadata["resolution_no"]).strip()

        # Generate summary page by page (2 pages at a time)
        pdf_file.seek(0)  # Reset file pointer
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        total_pages = len(doc)
        summaries = []
        
        for i in range(0, total_pages, 2):
            page_text = "".join(
                doc[j].get_text() for j in range(i, min(i + 2, total_pages))
            )
            summary = generate_summary(page_text, groq_client)
            summaries.append(summary)
        
        doc.close()

        # Combine summaries
        metadata["summary"] = "\n".join(summaries)
        metadata["filename"] = original_filename
        
        # Store in Pinecone permanent namespace using original filename
        vector_id = store_in_pinecone_permanent(metadata, original_filename, openai_client, pinecone_index)
        metadata["vector_id"] = vector_id
        
        # Clean up temporary namespace
        cleanup_temp_namespace(pinecone_index, temp_namespace)

        return metadata
        
    except Exception as e:
        # Clean up temporary namespace even if processing fails
        logging.error(f"Error during processing: {e}")
        cleanup_temp_namespace(pinecone_index, temp_namespace)
        raise

# Metadata extraction system prompt
METADATA_SYSTEM_PROMPT = """
Extract comprehensive metadata from given Security Council document strictly in the following structured format for efficient embedding and integration into the RAG system.

CRITICAL INSTRUCTION: Return ONLY ONE JSON object for the PRIMARY resolution in the document. If the document contains multiple resolutions, extract metadata for the FIRST or MOST PROMINENT resolution only.

Provide only the metadata fields exactly as specified below without any additional content, explanations, or multiple JSON objects:

{
  "resolution_no": "",  // Use case: Identify and retrieve specific resolutions by their unique identifier. IMPORTANT: Extract ONLY the resolution number (e.g., "20", "1234", "2758") WITHOUT the year in brackets. Do NOT include "(YYYY)" format.
  "year": "",  // Use case: Retrieve documents based on the year they were passed. Should be a number (e.g., 1947, 2023).
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

@upload_resolution_bp.route('/upload', methods=['POST'])
def upload_resolution():
    """
    API endpoint to upload and process a single PDF resolution.
    Uploads PDF to Supabase storage and processes in background.
    
    Request:
        - file: PDF file (multipart/form-data)
        
    Response:
        - JSON containing task_id for tracking background processing
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is PDF
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are allowed'}), 400
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Generate unique filename for storage
        unique_filename = f"{uuid4().hex}_{filename}"
        
        logging.info(f"Uploading PDF to Supabase storage: {unique_filename}")
        
        # Read file content
        file.seek(0)  # Reset file pointer
        file_content = file.read()
        
        # Upload to Supabase storage bucket
        try:
            storage_response = current_app.supabase_admin.storage.from_("audio_bucket").upload(
                path=unique_filename,
                file=file_content,
                file_options={"content-type": "application/pdf"}
            )
            logging.info(f"Storage response: {storage_response}")
        except Exception as storage_error:
            logging.error(f"Storage upload error: {str(storage_error)}")
            # Try with regular client as fallback
            try:
                storage_response = current_app.supabase.storage.from_("audio_bucket").upload(
                    path=unique_filename,
                    file=file_content,
                    file_options={"content-type": "application/pdf"}
                )
                logging.info(f"Fallback storage response: {storage_response}")
            except Exception as fallback_error:
                logging.error(f"Fallback storage error: {str(fallback_error)}")
                return jsonify({"error": f"Failed to upload to storage: {str(fallback_error)}"}), 500
        
        # Get the public URL for the uploaded file
        try:
            storage_url = current_app.supabase_admin.storage.from_("audio_bucket").get_public_url(unique_filename)
            logging.info(f"PDF file uploaded successfully: {storage_url}")
        except Exception as url_error:
            logging.error(f"Error getting public URL: {str(url_error)}")
            # Try to get signed URL as fallback
            try:
                storage_url = current_app.supabase_admin.storage.from_("audio_bucket").create_signed_url(unique_filename, 3600)  # 1 hour expiry
                if isinstance(storage_url, dict) and 'signedURL' in storage_url:
                    storage_url = storage_url['signedURL']
                logging.info(f"Using signed URL: {storage_url}")
            except Exception as signed_url_error:
                logging.error(f"Error getting signed URL: {str(signed_url_error)}")
                return jsonify({"error": "Failed to get file URL"}), 500
        
        # Import and enqueue Celery task with storage URL
        from tasks.transcribe_tasks import process_resolution_task
        task = process_resolution_task.delay(unique_filename, storage_url)
        
        logging.info(f"PDF processing task enqueued: {unique_filename}, task_id: {task.id}")
        
        return jsonify({"task_id": task.id}), 202, {"Location": f"/tasks/{task.id}"}
        
    except ValueError as ve:
        logging.error(f"Value error: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"Error uploading PDF: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to upload PDF: {str(e)}'}), 500

@upload_resolution_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

