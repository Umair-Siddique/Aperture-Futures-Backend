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
from math import ceil
from openai import OpenAI
from config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .auth import token_required
import imageio_ffmpeg
from report.generate_report import generate_and_store_transcription_report
import gc
from tasks.transcribe_tasks import transcribe_audio_task
from celery.result import AsyncResult
from celery_app import celery_app

from concurrent.futures import ThreadPoolExecutor, as_completed

import psutil
import logging

# Import shared utilities
from transcription_utils import (
    transcribe_large_audio_optimized,
    transcribe_audio_with_openai,
    preprocess_and_chunk,
    sanitize_id,
    batch_embed_and_upsert_optimized,
    download_audio,
    convert_and_compress_audio_optimized
)

transcribe_bp = Blueprint('transcribe', __name__)

# Initialize OpenAI client with error handling
try:
    if not Config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None


CHUNK_SIZE = 8192  # 8KB chunks for streaming
CHUNK_DURATION_SECONDS = 60   # 1 minute chunks
MAX_CONCURRENT_CHUNKS = 10     # threads

def check_memory_usage():
    """Check current memory usage and trigger cleanup if needed."""
    memory_percent = psutil.virtual_memory().percent
    # Removed memory threshold check - let system handle memory naturally
    current_app.logger.info(f"Current memory usage: {memory_percent}%")
    gc.collect()
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

@transcribe_bp.route("/audio", methods=["POST"])
@token_required
def transcribe_and_store(user):
    title = request.form.get("title")
    description = request.form.get("description")
    members_raw = request.form.get("members")

    if not title or not description or not members_raw:
        return jsonify({"error": "title, description, and members are required"}), 400

    members_list = [m.strip() for m in members_raw.split(",") if m.strip()]
    if not members_list:
        return jsonify({"error": "members cannot be empty"}), 400

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio file selected"}), 400

    filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    audio_file.save(filepath)

    # enqueue Celery task
    task = transcribe_audio_task.delay(title, description, members_list, filepath)

    return jsonify({"task_id": task.id}), 202, {"Location": f"/tasks/{task.id}"}

@transcribe_bp.route("/tasks/<task_id>", methods=["GET"])
@token_required  # Add authentication
def task_status(user, task_id):
    """Get the status of a background transcription task."""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state == 'PENDING':
            # Task is waiting to be processed
            response = {
                'id': task_id,
                'state': result.state,
                'current': 0,
                'total': 100,
                'status': 'Task is waiting to be processed...'
            }
        elif result.state == 'PROGRESS':
            # Task is currently being processed
            response = {
                'id': task_id,
                'state': result.state,
                'current': result.info.get('current', 0),
                'total': result.info.get('total', 100),
                'status': result.info.get('status', 'Processing...')
            }
        elif result.state == 'SUCCESS':
            # Task completed successfully
            response = {
                'id': task_id,
                'state': result.state,
                'current': 100,
                'total': 100,
                'status': 'Task completed successfully',
                'result': result.result
            }
        elif result.state == 'FAILURE':
            # Task failed
            response = {
                'id': task_id,
                'state': result.state,
                'current': 0,
                'total': 100,
                'status': 'Task failed',
                'error': str(result.info.get('error', 'Unknown error'))
            }
        else:
            # Unknown state
            response = {
                'id': task_id,
                'state': result.state,
                'status': f'Unknown task state: {result.state}'
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f"Error checking task status: {str(e)}")
        return jsonify({
            'id': task_id,
            'state': 'ERROR',
            'error': 'Failed to check task status'
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
                .select("title, description, timestamp", count="exact")  #  include description
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

        # Report generation - directly added
        report_info = {}
        try:
            report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
            current_app.logger.info(f"Report generation completed for: {title}")
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