from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
import whisper
import uuid
import tempfile
import re
import unicodedata
import subprocess
import requests
import yt_dlp as youtube_dl
from urllib.parse import urlparse
from datetime import datetime
from math import ceil
from openai import OpenAI
from config import Config
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .auth import token_required

transcribe_bp = Blueprint('transcribe', __name__)
client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Helper: Chunk Text Using RecursiveCharacterTextSplitter
def preprocess_and_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

# --- TRANSCRIBE AND STORE ENDPOINT ---
import re
import unicodedata

def sanitize_id(name: str) -> str:
    """Sanitize string for Pinecone namespace/vector IDs (ASCII only)."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', normalized)
    return sanitized.lower()


@transcribe_bp.route("/audio", methods=["POST"])
@token_required
def transcribe_and_store(user):  
    title = request.form.get("title")
    description = request.form.get("description")
    members_raw = request.form.get("members")  # comma-separated list from client
    audio_file = request.files.get("audio")

    # --- Validation ---
    if not title or not audio_file or not description or not members_raw:
        return jsonify({"error": "title, description, members, and audio file required"}), 400

    # Parse members into a Python list (for Supabase text[])
    members_list = [m.strip() for m in members_raw.split(",") if m.strip()]
    if not members_list:
        return jsonify({"error": "members cannot be empty"}), 400

    # Check for duplicate title
    response = current_app.supabase.table('audio_files').select('title').eq('title', title).execute()
    if response.data:
        return jsonify({"error": f"title '{title}' already exists"}), 409

    # Save audio file temporarily
    filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    audio_file.save(filepath)

    timestamp = int(time.time())

    # Store metadata in Supabase (includes description + members, NOT transcript)
    current_app.supabase.table('audio_files').insert({
        "title": title,
        "description": description,
        "members": members_list,   # Supabase maps this to text[]
        "timestamp": timestamp
    }).execute()

    # Transcribe with Whisper
    model = whisper.load_model("base")
    result = model.transcribe(filepath)
    transcript = result.get("text", "")

    if not transcript:
        os.remove(filepath)
        return jsonify({"error": "Transcription failed"}), 500

    # Split transcript into chunks
    chunks = preprocess_and_chunk(transcript)

    # Safe namespace for Pinecone
    safe_namespace = sanitize_id(title)

    # Embed + upsert into Pinecone
    def batch_embed_and_upsert(chunks, batch_size=16):
        total_chunks = len(chunks)
        vectors = []
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
            vectors.extend([
                (
                    f"{safe_namespace}_{i+j}",  # safe vector ID
                    vec,
                    {"text": chunk}             # ONLY transcript text stored
                )
                for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks))
            ])
        current_app.pinecone_index.upsert(vectors=vectors, namespace=safe_namespace)
        return len(vectors)

    chunks_stored = batch_embed_and_upsert(chunks, batch_size=16)

    # Cleanup temp file
    os.remove(filepath)

    return jsonify({
        "title": title,
        "description": description,
        "members": members_list,
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
    safe_namespace = sanitize_id(title)

    result = current_app.pinecone_index.query(
        vector=query_embedding,
        top_k=7,
        namespace=safe_namespace,
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
        "items": [
          {"title": "...", "description": "...", "timestamp": 1234567890}, ...
        ],
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
                .select("title, description, timestamp", count="exact")  # 👈 include description
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
    

@transcribe_bp.route("/delete", methods=["DELETE"])
@token_required
def delete_transcription(user):
    title = request.args.get("title")
    if not title:
        return jsonify({"error": "Title is required"}), 400

    safe_title = sanitize_id(title)

    try:
        # 1. Delete from Supabase (raw title)
        resp = (
            current_app.supabase
            .table("audio_files")
            .delete()
            .eq("title", title)   # raw title, since that's how it was stored
            .execute()
        )

        if not resp.data:
            return jsonify({"error": f"No transcription found with title '{title}'"}), 404

        # 2. Delete entire namespace from Pinecone
        try:
            current_app.pinecone_index.delete(
                namespace=title,
                delete_all=True
            )
        except Exception as e:
            current_app.logger.error("Pinecone delete error: %s", e)
            return jsonify({"error": "Failed to delete from Pinecone"}), 500

        return jsonify({"message": f"Transcription '{title}' deleted successfully"}), 200

    except Exception as e:
        current_app.logger.error("Delete error: %s", e)
        return jsonify({"error": "Failed to delete transcription"}), 500


# -------------------- /video Endpoint --------------------
def find_stream_url_from_un(page_url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(page_url, headers=headers, timeout=20)
        resp.raise_for_status()
        html = resp.text

        kaltura_pattern = r"(https://cdnapisec\.kaltura\.com/p/[\w/]+/playManifest/entryId/([\w_]+)/.*?\.m3u8\?[\w=&:;-]+)"
        m = re.search(kaltura_pattern, html)
        if m:
            return m.group(1), f"un_video_{m.group(2)}"

        jw_pattern = r"https://cdn\.jwplayer\.com/manifests/(\w+)\.m3u8"
        m = re.search(jw_pattern, html)
        if m:
            return m.group(0), f"un_video_{m.group(1)}"

        return None, None
    except requests.exceptions.RequestException:
        return None, None


def download_audio(url: str):
    """Download best audio track to temp dir via yt_dlp; returns (path, title)."""
    final_url = url
    video_title = "video_audio"

    if "webtv.un.org" in url:
        stream_url, generated_title = find_stream_url_from_un(url)
        if not stream_url:
            return None, None
        final_url = stream_url
        video_title = generated_title

    tmpdir = tempfile.gettempdir()
    unique = uuid.uuid4().hex
    sanitized_title = sanitize_id(video_title)
    outtmpl = os.path.join(tmpdir, f"{unique}_{sanitized_title}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "nocheckcertificate": True,
        "quiet": True,
        "outtmpl": outtmpl,
        "downloader": "ffmpeg",
        "hls_use_mpegts": True,
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(final_url, download=True)
            audio_path = ydl.prepare_filename(info)  # actual downloaded file path
        return (audio_path if os.path.exists(audio_path) else None), video_title
    except Exception:
        return None, None

def convert_and_compress_audio(input_audio_path: str):
    """Convert to small MP3 for cheaper/faster transcription."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        return None
    output_audio_path = os.path.splitext(input_audio_path)[0] + ".mp3"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_audio_path,
                "-acodec", "libmp3lame",
                "-ab", "64k",
                "-ar", "22050",
                output_audio_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.exists(output_audio_path):
            try:
                os.remove(input_audio_path)
            except Exception:
                pass
            return output_audio_path
        return None
    except subprocess.CalledProcessError:
        return None

def transcribe_audio_with_openai(audio_path: str):
    """Transcribe with OpenAI Whisper-1."""
    if not audio_path or not os.path.exists(audio_path):
        return None
    try:
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return getattr(resp, "text", None)
    except Exception as e:
        current_app.logger.error("Transcription error: %s", e)
        return None



@transcribe_bp.route("/video", methods=["POST"])
@token_required
def transcribe_video_and_store(user):
    title = request.form.get("title")
    description = request.form.get("description")
    video_url = request.form.get("url") or request.form.get("video_url")
    members_raw = request.form.get("members")

    if not title or not description or not video_url or not members_raw:
        return jsonify({"error": "title, description, url, and members are required"}), 400

    members_list = [m.strip() for m in members_raw.split(",") if m.strip()]
    if not members_list:
        return jsonify({"error": "members cannot be empty"}), 400

    parsed = urlparse(video_url)
    if not (parsed.scheme and parsed.netloc):
        return jsonify({"error": "Invalid URL"}), 400

    exists = current_app.supabase.table("audio_files").select("title").eq("title", title).execute()
    if exists.data:
        return jsonify({"error": f"title '{title}' already exists"}), 409

    now_epoch = int(time.time())

    try:
        current_app.supabase.table("audio_files").insert({
            "title": title,
            "description": description,
            "members": members_list,
            "timestamp": now_epoch,
        }).execute()
    except Exception as e:
        current_app.logger.error("Supabase insert error: %s", e)
        return jsonify({"error": "Failed to insert record"}), 500

    tmp_mp3 = None
    try:
        audio_path, inferred_title = download_audio(video_url)
        if not audio_path:
            return jsonify({"error": "Audio download failed"}), 500

        tmp_mp3 = convert_and_compress_audio(audio_path)
        if not tmp_mp3:
            return jsonify({"error": "Audio conversion/compression failed"}), 500

        transcript = transcribe_audio_with_openai(tmp_mp3)
        if not transcript:
            return jsonify({"error": "Transcription failed"}), 500

        chunks = preprocess_and_chunk(transcript)

        def batch_embed_and_upsert(chunks_list, batch_size=16):
            safe_namespace = sanitize_id(title)
            total = 0
            for i in range(0, len(chunks_list), batch_size):
                batch_chunks = chunks_list[i:i + batch_size]
                batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
                vectors = []
                for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks)):
                    vectors.append((
                        f"{safe_namespace}_{i + j}",
                        vec,
                        {"text": chunk},
                    ))
                current_app.pinecone_index.upsert(vectors=vectors, namespace=safe_namespace)
                total += len(vectors)
            return total

        chunks_stored = batch_embed_and_upsert(chunks, batch_size=16)

        return jsonify({
            "title": title,
            "description": description,
            "members": members_list,
            "url": video_url,
            "timestamp": now_epoch,
            "chunks_stored": chunks_stored,
        }), 200

    finally:
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass