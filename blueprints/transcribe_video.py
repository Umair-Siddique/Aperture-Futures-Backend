from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import re
import uuid
import tempfile
import time
from datetime import datetime
import subprocess
import requests
import yt_dlp as youtube_dl
from urllib.parse import urlparse
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import Config
from .auth import token_required

# ---- Blueprint ----
transcribe_video_bp = Blueprint("transcribe_video", __name__)
# transcribe_bp = Blueprint('transcribe', __name__)

# ---- OpenAI client for Whisper-1 ----
client = OpenAI(api_key=Config.OPENAI_API_KEY)

# ---- Helpers ----
def preprocess_and_chunk(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return text_splitter.split_text(text or "")

def sanitize_filename(filename: str):
    sanitized = re.sub(r"[^\w\s-]", "", filename).strip()
    sanitized = re.sub(r"[-\s]+", "-", sanitized)
    return sanitized

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
    sanitized_title = sanitize_filename(video_title)
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

import re
import unicodedata

def sanitize_id(name: str) -> str:
    """Sanitize string for Pinecone namespace/vector IDs (ASCII only)."""
    normalized = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', normalized)
    return sanitized.lower()

# ---- VIDEO -> TRANSCRIBE -> EMBED ----
@transcribe_video_bp.route("/video", methods=["POST"])
@token_required
def transcribe_video_and_store(user):
    """
    Form-data:
      - title (str)        REQUIRED (must be unique)
      - description (str)  REQUIRED
      - url (str)          REQUIRED (video URL)
      - members (str)      REQUIRED (comma-separated list)
    Returns:
      { "title", "description", "members", "url", "timestamp", "chunks_stored" }
    """
    title = request.form.get("title")
    description = request.form.get("description")
    video_url = request.form.get("url") or request.form.get("video_url")
    members_raw = request.form.get("members")

    # --- Validation ---
    if not title or not description or not video_url or not members_raw:
        return jsonify({"error": "title, description, url, and members are required"}), 400

    # Parse members into list
    members_list = [m.strip() for m in members_raw.split(",") if m.strip()]
    if not members_list:
        return jsonify({"error": "members cannot be empty"}), 400

    # Ensure URL is valid-ish
    parsed = urlparse(video_url)
    if not (parsed.scheme and parsed.netloc):
        return jsonify({"error": "Invalid URL"}), 400

    # Enforce unique title (same behavior as /audio)
    exists = current_app.supabase.table("audio_files").select("title").eq("title", title).execute()
    if exists.data:
        return jsonify({"error": f"title '{title}' already exists"}), 409

    now_epoch = int(time.time())

    # --- Store metadata in Supabase (NO transcript here) ---
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
        # 1) Download audio
        audio_path, inferred_title = download_audio(video_url)
        if not audio_path:
            return jsonify({"error": "Audio download failed"}), 500

        # 2) Convert/compress
        tmp_mp3 = convert_and_compress_audio(audio_path)
        if not tmp_mp3:
            return jsonify({"error": "Audio conversion/compression failed"}), 500

        # 3) Transcribe
        transcript = transcribe_audio_with_openai(tmp_mp3)
        if not transcript:
            return jsonify({"error": "Transcription failed"}), 500

        # 4) Chunk
        chunks = preprocess_and_chunk(transcript)

        # 5) Embed + upsert into Pinecone
        def batch_embed_and_upsert(chunks_list, batch_size=16):
            safe_namespace = sanitize_id(title)   # same safe ID logic from /audio
            total = 0
            for i in range(0, len(chunks_list), batch_size):
                batch_chunks = chunks_list[i:i + batch_size]
                batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
                vectors = []
                for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks)):
                    vectors.append((
                        f"{safe_namespace}_{i + j}",   # safe vector ID
                        vec,
                        {"text": chunk},               # ONLY transcript text
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
        # cleanup temp file
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass
