from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import whisper
import uuid
import tempfile
from config import Config
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from flask import Blueprint, request, jsonify, current_app
from .auth import token_required
from datetime import datetime
from math import ceil
from flask import request, jsonify

transcribe_bp = Blueprint('transcribe', __name__)

# Helper: Chunk Text Using RecursiveCharacterTextSplitter
def preprocess_and_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

# --- TRANSCRIBE AND STORE ENDPOINT ---
@transcribe_bp.route("/audio", methods=["POST"])
@token_required
def transcribe_and_store(user):  
    title = request.form.get("title")
    description = request.form.get("description")
    audio_file = request.files.get("audio")

    if not title or not audio_file or not description:
        return jsonify({"error": "title, description, and audio file required"}), 400

    response = current_app.supabase.table('audio_files').select('title').eq('title', title).execute()
    if response.data:
        return jsonify({"error": f"title '{title}' already exists"}), 409

    filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    audio_file.save(filepath)

    timestamp = int(time.time())
    
    current_app.supabase.table('audio_files').insert({
    "title": title,
    "description": description,
    "timestamp": timestamp
}).execute()


    model = whisper.load_model("base")
    result = model.transcribe(filepath)
    transcript = result.get("text", "")

    if not transcript:
        os.remove(filepath)
        return jsonify({"error": "Transcription failed"}), 500

    chunks = preprocess_and_chunk(transcript)

    def batch_embed_and_upsert(chunks, batch_size=16):
        total_chunks = len(chunks)
        vectors = []
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
            vectors.extend([
                (
                    f"{title}_{i+j}",
                    vec,
                    {"text": chunk, "description": description}
                )
                for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks))
            ])
        current_app.pinecone_index.upsert(vectors=vectors, namespace=title)
        return len(vectors)

    chunks_stored = batch_embed_and_upsert(chunks, batch_size=16)

    os.remove(filepath)

    return jsonify({
        "title": title,
        "description": description,
        "timestamp": timestamp,
        "chunks_stored": chunks_stored
    })

# --- RETRIEVE AND GENERATE ENDPOINT ---
@transcribe_bp.route("/retrieve", methods=["POST"])
@token_required
def retrieve_transcript(user):
    query = request.form.get("query")
    title = request.form.get("title")

    if not query or not title:
        return jsonify({"error": "Both 'query' and 'title' are required"}), 400

    record = current_app.supabase.table('audio_files').select('description').eq('title', title).execute()
    if not record.data:
        return jsonify({"error": f"No record found for title '{title}'"}), 404

    description = record.data[0]['description']

    query_embedding = current_app.embeddings.embed_query(query)
    result = current_app.pinecone_index.query(
        vector=query_embedding,
        top_k=7,
        namespace=title,
        include_metadata=True
    )

    contexts = [match['metadata']['text'] for match in result['matches']]
    combined_context = "\n".join(contexts)

    llm = ChatOpenAI(
        openai_api_key=Config.OPENAI_API_KEY,
        model_name="gpt-3.5-turbo"
    )

    rag_prompt = (
        "You are a helpful assistant. Using only the following transcript chunks and the provided meeting description, answer the user's question.\n\n"
        f"Meeting Description: {description}\n"
        f"Transcript:\n{combined_context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": rag_prompt}
    ])
    answer = response.content if hasattr(response, 'content') else response['content']

    return jsonify({
        "query": query,
        "description": description,
        "context_chunks": contexts,
        "answer": answer
    })

@transcribe_bp.route("/list-transcription", methods=["GET"])
@token_required
def list_audio_files(user):
    """
    Paginated list of audio files (fixed 10 per page).
    Query params:
      - page: 1-based page number (default 1)
    Response:
      {
        "items": [{"title": "...", "timestamp": 1234567890}, ...],
        "page": 1,
        "limit": 10,
        "total": 42,
        "total_pages": 5,
        "has_next": true,
        "has_prev": false
      }
    """
    try:
        # Parse and clamp page
        try:
            page = int(request.args.get("page", 1))
        except (TypeError, ValueError):
            page = 1
        page = max(page, 1)

        limit = 10
        start = (page - 1) * limit
        end = start + limit - 1  # inclusive end for Supabase .range()

        resp = (
            current_app.supabase
                .table("audio_files")
                .select("title, timestamp", count="exact")
                .order("timestamp", desc=True)
                .order("title", desc=True)  # tie-breaker for stable ordering
                .range(start, end)
                .execute()
        )

        items = resp.data or []
        total = resp.count or 0
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
        current_app.logger.error("Supabase list error: %s", e)
        return jsonify({"error": "Failed to list transcriptions"}), 500