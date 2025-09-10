from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os
import time
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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .auth import token_required
import librosa
import soundfile as sf
import imageio_ffmpeg
from report.generate_report import generate_and_store_transcription_report
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import psutil

transcribe_bp = Blueprint('transcribe', __name__)

# Initialize OpenAI client with error handling
try:
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB max upload
CHUNK_SIZE = 8192  # 8KB chunks for streaming
CHUNK_DURATION_SECONDS = 30  # Reduced from 60 to 30 seconds for better memory management
MAX_CONCURRENT_CHUNKS = 3  # Limit concurrent processing
MEMORY_THRESHOLD_PERCENT = 80  # Memory usage threshold for cleanup

def check_memory_usage():
    """Check current memory usage and trigger cleanup if needed."""
    memory_percent = psutil.virtual_memory().percent
    if memory_percent > MEMORY_THRESHOLD_PERCENT:
        current_app.logger.warning(f"High memory usage: {memory_percent}%")
        gc.collect()
        return True
    return False

def save_uploaded_file_streaming(file_storage: FileStorage, target_path: str) -> bool:
    """Save uploaded file using streaming to handle large files efficiently."""
    try:
        with open(target_path, 'wb') as f:
            while True:
                chunk = file_storage.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as e:
        current_app.logger.error(f"Error saving file: {str(e)}")
        return False

def split_audio_into_chunks_optimized(input_path: str, max_size=MAX_FILE_SIZE):
    """Optimized audio chunking with memory management."""
    current_app.logger.info(f"Starting audio chunking for: {input_path}")
    
    try:
        # Check if input file exists
        if not os.path.exists(input_path):
            current_app.logger.error(f"Input file does not exist: {input_path}")
            return []
        
        # Load audio with librosa using lower sample rate for memory efficiency
        current_app.logger.info("Loading audio with librosa...")
        audio, sr = librosa.load(input_path, sr=16000)  # Reduced sample rate
        duration_samples = len(audio)
        duration_seconds = duration_samples / sr
        
        current_app.logger.info(f"Audio loaded: {duration_seconds:.2f} seconds, {sr}Hz sample rate")

        chunks = []
        start_sample = 0
        chunk_count = 0
        
        while start_sample < duration_samples:
            chunk_count += 1
            # Start with smaller chunks for better memory management
            chunk_duration_seconds = CHUNK_DURATION_SECONDS
            end_sample = min(start_sample + int(chunk_duration_seconds * sr), duration_samples)
            
            # Extract chunk
            chunk_audio = audio[start_sample:end_sample]
            
            # Save chunk to temporary file with compression
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                # Use lower bit depth for smaller file size
                sf.write(tmpf.name, chunk_audio, sr, subtype='PCM_16')
                size = os.path.getsize(tmpf.name)
                
                # If chunk is still too large, reduce duration progressively
                while size > max_size and chunk_duration_seconds > 5:
                    chunk_duration_seconds -= 5
                    end_sample = start_sample + int(chunk_duration_seconds * sr)
                    chunk_audio = audio[start_sample:end_sample]
                    
                    # Rewrite with smaller chunk
                    sf.write(tmpf.name, chunk_audio, sr, subtype='PCM_16')
                    size = os.path.getsize(tmpf.name)
                
                chunks.append(tmpf.name)
                current_app.logger.info(f"Created chunk {chunk_count}: {size} bytes, {chunk_duration_seconds:.1f}s")
                start_sample = end_sample

        # Clear audio from memory
        del audio
        gc.collect()
        
        current_app.logger.info(f"Successfully created {len(chunks)} audio chunks")
        return chunks
        
    except Exception as e:
        current_app.logger.error(f"Error splitting audio with librosa: {str(e)}")
        current_app.logger.error(f"Error type: {type(e).__name__}")
        import traceback
        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def transcribe_audio_with_openai(audio_path: str):
    """Send audio to OpenAI Whisper for transcription with retry logic."""
    max_retries = 3
    current_app.logger.info(f"Starting transcription for file: {audio_path}")
    
    # Check if OpenAI client is available
    if not client:
        current_app.logger.error("OpenAI client not initialized - check API key")
        return ""
    
    # Check if file exists and get size
    if not os.path.exists(audio_path):
        current_app.logger.error(f"Audio file does not exist: {audio_path}")
        return ""
    
    file_size = os.path.getsize(audio_path)
    current_app.logger.info(f"Audio file size: {file_size} bytes")
    
    for attempt in range(max_retries):
        try:
            current_app.logger.info(f"Transcription attempt {attempt + 1}/{max_retries}")
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",  # Use whisper-1 for better performance
                    file=f
                )
            
            if transcript and transcript.text:
                current_app.logger.info(f"Transcription successful, text length: {len(transcript.text)}")
                return transcript.text
            else:
                current_app.logger.warning(f"Transcription returned empty result for {audio_path}")
                return ""
                
        except Exception as e:
            error_msg = f"Transcription attempt {attempt + 1} failed: {str(e)}"
            current_app.logger.error(error_msg)
            current_app.logger.error(f"Error type: {type(e).__name__}")
            
            if attempt == max_retries - 1:
                current_app.logger.error(f"All transcription attempts failed for {audio_path}")
                return ""
            time.sleep(2)  # Wait before retry
    
    return ""


def transcribe_large_audio_optimized(audio_path: str):
    """Optimized large audio transcription with concurrent processing and memory management."""
    current_app.logger.info(f"Starting large audio transcription for: {audio_path}")
    
    # Check if input file exists
    if not os.path.exists(audio_path):
        current_app.logger.error(f"Input audio file does not exist: {audio_path}")
        return ""
    
    input_file_size = os.path.getsize(audio_path)
    current_app.logger.info(f"Input file size: {input_file_size} bytes")
    
    try:
        all_chunks = split_audio_into_chunks_optimized(audio_path, MAX_FILE_SIZE)
        if not all_chunks:
            current_app.logger.error("Failed to split audio into chunks")
            return ""
        
        transcripts = []
        total_chunks = len(all_chunks)
        current_app.logger.info(f"Successfully created {total_chunks} audio chunks")
        
        # Process chunks in batches to manage memory
        for i in range(0, len(all_chunks), MAX_CONCURRENT_CHUNKS):
            batch_chunks = all_chunks[i:i + MAX_CONCURRENT_CHUNKS]
            batch_num = i // MAX_CONCURRENT_CHUNKS + 1
            total_batches = (total_chunks + MAX_CONCURRENT_CHUNKS - 1) // MAX_CONCURRENT_CHUNKS
            
            current_app.logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_chunks)} chunks")
            
            # Check memory usage before processing batch
            check_memory_usage()
            
            # Use ThreadPoolExecutor for concurrent processing
            with ThreadPoolExecutor(max_workers=min(len(batch_chunks), MAX_CONCURRENT_CHUNKS)) as executor:
                # Submit transcription tasks
                future_to_chunk = {
                    executor.submit(transcribe_audio_with_openai, chunk_path): chunk_path 
                    for chunk_path in batch_chunks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_path = future_to_chunk[future]
                    try:
                        text = future.result()
                        if text:
                            transcripts.append(text)
                            current_app.logger.info(f"Successfully transcribed chunk: {chunk_path}")
                        else:
                            current_app.logger.warning(f"Empty transcription for chunk: {chunk_path}")
                    except Exception as e:
                        current_app.logger.error(f"Error transcribing chunk {chunk_path}: {str(e)}")
                    finally:
                        # Cleanup chunk file
                        try:
                            os.remove(chunk_path)
                        except Exception as e:
                            current_app.logger.warning(f"Failed to cleanup chunk file {chunk_path}: {str(e)}")
            
            # Force garbage collection after each batch
            gc.collect()
            
            # Check memory usage after processing batch
            check_memory_usage()
        
        final_transcript = " ".join(transcripts)
        current_app.logger.info(f"Completed transcription of {total_chunks} chunks, final text length: {len(final_transcript)}")
        
        if not final_transcript.strip():
            current_app.logger.error("Final transcript is empty")
            return ""
            
        return final_transcript
        
    except Exception as e:
        current_app.logger.error(f"Error in transcribe_large_audio_optimized: {str(e)}")
        current_app.logger.error(f"Error type: {type(e).__name__}")
        import traceback
        current_app.logger.error(f"Traceback: {traceback.format_exc()}")
        return ""


# Helper: Chunk Text Using RecursiveCharacterTextSplitter
def preprocess_and_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)


def sanitize_id(name: str) -> str:
    """Sanitize string for Pinecone namespace/vector IDs (ASCII only)."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', normalized)
    return sanitized.lower()


def batch_embed_and_upsert_optimized(chunks, safe_namespace, batch_size=8):
    """Optimized embedding and upsert with memory management."""
    total_chunks = len(chunks)
    total_vectors = 0
    
    # Process in smaller batches to reduce memory usage
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        
        try:
            # Generate embeddings for this batch
            batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
            
            # Create vectors for this batch
            vectors = [
                (
                    f"{safe_namespace}_{i+j}",
                    vec,
                    {"text": chunk}
                )
                for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks))
            ]
            
            # Upsert this batch
            current_app.pinecone_index.upsert(vectors=vectors, namespace=safe_namespace)
            total_vectors += len(vectors)
            
            # Clear batch data from memory
            del batch_embeds, vectors
            gc.collect()
            
        except Exception as e:
            current_app.logger.error(f"Error processing embedding batch {i//batch_size + 1}: {str(e)}")
            continue
    
    return total_vectors


@transcribe_bp.route("/audio", methods=["POST"])
@token_required
def transcribe_and_store(user):
    # Get form data first (before file processing)
    title = request.form.get("title")
    description = request.form.get("description")
    members_raw = request.form.get("members")
    
    if not title or not description or not members_raw:
        return jsonify({"error": "title, description, and members are required"}), 400

    members_list = [m.strip() for m in members_raw.split(",") if m.strip()]
    if not members_list:
        return jsonify({"error": "members cannot be empty"}), 400

    # Check for file in request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400

    # Check file size before processing
    audio_file.seek(0, 2)  # Seek to end
    file_size = audio_file.tell()
    audio_file.seek(0)  # Reset to beginning
    
    if file_size > MAX_UPLOAD_SIZE:
        return jsonify({"error": f"File too large. Maximum size: {MAX_UPLOAD_SIZE // (1024*1024)}MB"}), 413

    # Check memory usage before starting
    check_memory_usage()

    # Save uploaded file using streaming
    filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    
    if not save_uploaded_file_streaming(audio_file, filepath):
        return jsonify({"error": "Failed to save uploaded file"}), 500

    try:
        # Insert metadata into Supabase
        timestamp = int(time.time())
        current_app.supabase.table('audio_files').insert({
            "title": title,
            "description": description,
            "members": members_list,
            "timestamp": timestamp
        }).execute()

        # Transcribe large file safely with optimized processing
        current_app.logger.info(f"Starting transcription for: {title}")
        transcript = transcribe_large_audio_optimized(filepath)

        if not transcript:
            current_app.logger.warning("Optimized transcription failed, trying fallback method...")
            # Try fallback method - direct transcription without chunking
            try:
                transcript = transcribe_audio_with_openai(filepath)
                if not transcript:
                    current_app.logger.error("Both optimized and fallback transcription failed")
                    # Clean up database entry if transcription failed
                    current_app.supabase.table('audio_files').delete().eq("title", title).execute()
                    return jsonify({"error": "Transcription failed - both optimized and fallback methods failed"}), 500
                else:
                    current_app.logger.info("Fallback transcription successful")
            except Exception as e:
                current_app.logger.error(f"Fallback transcription also failed: {str(e)}")
                # Clean up database entry if transcription failed
                current_app.supabase.table('audio_files').delete().eq("title", title).execute()
                return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

        # Report generation
        report_info = {}
        try:
            report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
        except Exception as e:
            current_app.logger.error("Report generation failed: %s", e)

        # Chunk transcript → embeddings with optimized processing
        current_app.logger.info(f"Processing embeddings for: {title}")
        chunks = preprocess_and_chunk(transcript)
        safe_namespace = sanitize_id(title)
        chunks_stored = batch_embed_and_upsert_optimized(chunks, safe_namespace, batch_size=8)

        # Final memory cleanup
        del transcript, chunks
        gc.collect()

        current_app.logger.info(f"Successfully completed transcription for: {title}")
        return jsonify({
            "title": title,
            "description": description,
            "members": members_list,
            "timestamp": timestamp,
            "chunks_stored": chunks_stored,
            "report_saved": bool(report_info.get("ok"))
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error in audio transcription: {str(e)}")
        # Clean up database entry on error
        try:
            current_app.supabase.table('audio_files').delete().eq("title", title).execute()
        except Exception:
            pass
        return jsonify({"error": "Internal server error during transcription"}), 500

    finally:
        # Always cleanup uploaded file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass

@transcribe_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to monitor system status."""
    try:
        memory_percent = psutil.virtual_memory().percent
        return jsonify({
            "status": "healthy",
            "memory_usage_percent": memory_percent,
            "max_upload_size_mb": MAX_UPLOAD_SIZE // (1024 * 1024),
            "timestamp": int(time.time())
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": int(time.time())
        }), 500

@transcribe_bp.route("/test-transcription", methods=["POST"])
@token_required
def test_transcription(user):
    """Test endpoint to debug transcription issues."""
    try:
        # Check if client is initialized
        if not client:
            return jsonify({
                "status": "error",
                "openai_working": False,
                "error": "OpenAI client not initialized - check API key configuration",
                "timestamp": int(time.time())
            }), 500
        
        # Check if test file exists
        test_file_path = "testing_audios/3431341.mp3"
        if not os.path.exists(test_file_path):
            return jsonify({
                "status": "error",
                "openai_working": False,
                "error": f"Test file not found: {test_file_path}",
                "timestamp": int(time.time())
            }), 500
        
        # Test OpenAI API connection
        with open(test_file_path, "rb") as f:
            test_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        
        return jsonify({
            "status": "success",
            "openai_working": True,
            "test_transcription_length": len(test_response.text) if test_response.text else 0,
            "api_key_configured": bool(Config.OPENAI_API_KEY),
            "timestamp": int(time.time())
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "openai_working": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "api_key_configured": bool(Config.OPENAI_API_KEY),
            "timestamp": int(time.time())
        }), 500

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

    safe_namespace = sanitize_id(title)

    try:
        # 1. Delete from Supabase (raw title, because that's what we stored there)
        resp = (
            current_app.supabase
            .table("audio_files")
            .delete()
            .eq("title", title)   # match the original stored title
            .execute()
        )

        if not resp.data:
            return jsonify({"error": f"No transcription found with title '{title}'"}), 404

        # 2. Delete from Pinecone (use sanitized namespace!)
        try:
            current_app.pinecone_index.delete(
                namespace=safe_namespace,
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
        # ✅ no need to set "downloader": "ffmpeg" explicitly
        "hls_use_mpegts": True,
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(final_url, download=True)
            audio_path = ydl.prepare_filename(info)  # actual downloaded file path
        return (audio_path if os.path.exists(audio_path) else None), video_title
    except Exception:
        return None, None


def convert_and_compress_audio_optimized(input_audio_path: str):
    """Optimized audio conversion with better compression and memory management."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        return None
    output_audio_path = os.path.splitext(input_audio_path)[0] + ".mp3"
    try:
        # More aggressive compression for better memory usage
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-i", input_audio_path,
                "-acodec", "libmp3lame",
                "-ab", "32k",  # Reduced from 64k to 32k
                "-ar", "16000",  # Reduced from 22050 to 16000
                "-ac", "1",  # Convert to mono
                "-af", "volume=0.8",  # Slight volume reduction
                output_audio_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,  # 5 minute timeout
        )
        if os.path.exists(output_audio_path):
            try:
                os.remove(input_audio_path)
            except Exception:
                pass
            return output_audio_path
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        current_app.logger.error(f"Audio conversion failed: {str(e)}")
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

    # Check memory usage before starting
    check_memory_usage()

    now_epoch = int(time.time())
    tmp_mp3 = None
    audio_path = None

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

    try:
        current_app.logger.info(f"Starting video processing for: {title}")
        
        # Download audio from video
        audio_path, inferred_title = download_audio(video_url)
        if not audio_path:
            # Clean up database entry if download failed
            current_app.supabase.table('audio_files').delete().eq("title", title).execute()
            return jsonify({"error": "Audio download failed"}), 500

        # Convert and compress audio
        tmp_mp3 = convert_and_compress_audio_optimized(audio_path)
        if not tmp_mp3:
            # Clean up database entry if conversion failed
            current_app.supabase.table('audio_files').delete().eq("title", title).execute()
            return jsonify({"error": "Audio conversion/compression failed"}), 500

        # Clean up original audio file
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        # Optimized batch transcription
        current_app.logger.info(f"Starting transcription for: {title}")
        transcript = transcribe_large_audio_optimized(tmp_mp3)
        if not transcript:
            # Clean up database entry if transcription failed
            current_app.supabase.table('audio_files').delete().eq("title", title).execute()
            return jsonify({"error": "Transcription failed"}), 500

        # Report generation
        report_info = {}
        try:
            report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
        except Exception as e:
            current_app.logger.error("Report generation failed: %s", e)

        # Process embeddings
        current_app.logger.info(f"Processing embeddings for: {title}")
        chunks = preprocess_and_chunk(transcript)
        safe_namespace = sanitize_id(title)
        chunks_stored = batch_embed_and_upsert_optimized(chunks, safe_namespace, batch_size=8)

        # Final memory cleanup
        del transcript, chunks
        gc.collect()

        current_app.logger.info(f"Successfully completed video transcription for: {title}")
        return jsonify({
            "title": title,
            "description": description,
            "members": members_list,
            "url": video_url,
            "timestamp": now_epoch,
            "chunks_stored": chunks_stored,
            "report_saved": bool(report_info.get("ok"))
        }), 200

    except Exception as e:
        current_app.logger.error(f"Error in video transcription: {str(e)}")
        # Clean up database entry on error
        try:
            current_app.supabase.table('audio_files').delete().eq("title", title).execute()
        except Exception:
            pass
        return jsonify({"error": "Internal server error during video transcription"}), 500

    finally:
        # Always cleanup temporary files
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass
        try:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass