from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import time
# import whisper  # Remove this import
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
from pydub import AudioSegment
import imageio_ffmpeg
from report.generate_report import generate_and_store_transcription_report

transcribe_bp = Blueprint('transcribe', __name__)
client = OpenAI(api_key=Config.OPENAI_API_KEY)


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes

AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()


def split_audio_into_chunks(input_path: str, max_size=MAX_FILE_SIZE):
    """Split audio into chunks each ≤ max_size bytes."""
    audio = AudioSegment.from_file(input_path)
    duration_ms = len(audio)

    chunks = []
    start = 0
    while start < duration_ms:
        # progressively increase until size is too large
        end = duration_ms
        step = 60 * 1000  # start with 1 min increments
        while True:
            candidate = audio[start:end]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpf:
                candidate.export(tmpf.name, format="mp3")
                size = os.path.getsize(tmpf.name)
            if size <= max_size or end - start <= step:
                chunks.append(tmpf.name)
                start = end
                break
            else:
                # shrink by 1 minute
                end -= step

    return chunks


def transcribe_audio_with_openai(audio_path: str):
    """Send audio to OpenAI Whisper for transcription."""
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return transcript.text if transcript else ""


def transcribe_large_audio(audio_path: str):
    """Split large audio into chunks and transcribe each with Whisper."""
    all_chunks = split_audio_into_chunks(audio_path, MAX_FILE_SIZE)
    transcripts = []
    for chunk_path in all_chunks:
        text = transcribe_audio_with_openai(chunk_path)
        if text:
            transcripts.append(text)
        os.remove(chunk_path)  # cleanup each chunk
    return " ".join(transcripts)


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


def batch_embed_and_upsert(chunks, safe_namespace, batch_size=16):
    """Embed transcript chunks and upsert them into Pinecone."""
    total_chunks = len(chunks)
    vectors = []
    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
        vectors.extend([
            (
                f"{safe_namespace}_{i+j}",
                vec,
                {"text": chunk}
            )
            for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks))
        ])
    current_app.pinecone_index.upsert(vectors=vectors, namespace=safe_namespace)
    return len(vectors)


@transcribe_bp.route("/audio", methods=["POST"])
@token_required
def transcribe_and_store(user):
    title = request.form.get("title")
    description = request.form.get("description")
    members_raw = request.form.get("members")
    audio_file = request.files.get("audio")

    if not title or not audio_file or not description or not members_raw:
        return jsonify({"error": "title, description, members, and audio file required"}), 400

    members_list = [m.strip() for m in members_raw.split(",") if m.strip()]
    if not members_list:
        return jsonify({"error": "members cannot be empty"}), 400

    # Save uploaded file temporarily
    filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    audio_file.save(filepath)

    # Insert metadata into Supabase
    timestamp = int(time.time())
    current_app.supabase.table('audio_files').insert({
        "title": title,
        "description": description,
        "members": members_list,
        "timestamp": timestamp
    }).execute()

    # Transcribe large file safely
    transcript = transcribe_large_audio(filepath)

    os.remove(filepath)  # cleanup uploaded file

    if not transcript:
        return jsonify({"error": "Transcription failed"}), 500

    # Report generation
    report_info = {}
    try:
        report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
    except Exception as e:
        current_app.logger.error("Report generation failed: %s", e)

    # Chunk transcript → embeddings
    chunks = preprocess_and_chunk(transcript)
    safe_namespace = sanitize_id(title)
    chunks_stored = batch_embed_and_upsert(chunks, safe_namespace, batch_size=16)

    return jsonify({
        "title": title,
        "description": description,
        "members": members_list,
        "timestamp": timestamp,
        "chunks_stored": chunks_stored,
        "report_saved": bool(report_info.get("ok"))
    }), 200

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


def convert_and_compress_audio(input_audio_path: str):
    """Convert to small MP3 for cheaper/faster transcription using imageio-ffmpeg."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        return None
    output_audio_path = os.path.splitext(input_audio_path)[0] + ".mp3"
    try:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),  # ✅ use bundled ffmpeg
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

        # ✅ Batch transcription
        transcript = transcribe_large_audio(tmp_mp3)
        if not transcript:
            return jsonify({"error": "Transcription failed"}), 500

        # Report generation
        report_info = {}
        try:
            report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
        except Exception as e:
            current_app.logger.error("Report generation failed: %s", e)

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
            "report_saved": bool(report_info.get("ok"))
        }), 200

    finally:
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass