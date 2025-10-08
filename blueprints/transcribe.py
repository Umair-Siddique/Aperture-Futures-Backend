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
import json
from flask import Response, stream_with_context

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

    try:
        # Generate unique filename for storage
        filename = f"{uuid.uuid4().hex}_{secure_filename(audio_file.filename)}"
        
        # Upload to Supabase storage
        current_app.logger.info(f"Uploading audio file to Supabase storage: {filename}")
        
        # Read file content
        audio_file.seek(0)  # Reset file pointer
        file_content = audio_file.read()
        
        # Determine content type based on file extension
        file_ext = os.path.splitext(audio_file.filename)[1].lower()
        content_type_map = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg'
        }
        content_type = content_type_map.get(file_ext, 'audio/mpeg')
        
        # Upload to Supabase storage bucket using admin client for better permissions
        try:
            storage_response = current_app.supabase_admin.storage.from_("audio_bucket").upload(
                path=filename,
                file=file_content,
                file_options={"content-type": content_type}
            )
            current_app.logger.info(f"Storage response: {storage_response}")
        except Exception as storage_error:
            current_app.logger.error(f"Storage upload error: {str(storage_error)}")
            # Try with regular client as fallback
            try:
                storage_response = current_app.supabase.storage.from_("audio_bucket").upload(
                    path=filename,
                    file=file_content,
                    file_options={"content-type": content_type}
                )
                current_app.logger.info(f"Fallback storage response: {storage_response}")
            except Exception as fallback_error:
                current_app.logger.error(f"Fallback storage error: {str(fallback_error)}")
                return jsonify({"error": f"Failed to upload to storage: {str(fallback_error)}"}), 500
        
        # Get the public URL for the uploaded file
        try:
            storage_url = current_app.supabase_admin.storage.from_("audio_bucket").get_public_url(filename)
            current_app.logger.info(f"Audio file uploaded successfully: {storage_url}")
        except Exception as url_error:
            current_app.logger.error(f"Error getting public URL: {str(url_error)}")
            # Try to get signed URL as fallback
            try:
                storage_url = current_app.supabase_admin.storage.from_("audio_bucket").create_signed_url(filename, 3600)  # 1 hour expiry
                if isinstance(storage_url, dict) and 'signedURL' in storage_url:
                    storage_url = storage_url['signedURL']
                current_app.logger.info(f"Using signed URL: {storage_url}")
            except Exception as signed_url_error:
                current_app.logger.error(f"Error getting signed URL: {str(signed_url_error)}")
                return jsonify({"error": "Failed to get file URL"}), 500
        
        # enqueue Celery task with storage URL instead of local file path
        task = transcribe_audio_task.delay(title, description, members_list, storage_url, filename)

        return jsonify({"task_id": task.id}), 202, {"Location": f"/tasks/{task.id}"}
        
    except Exception as e:
        current_app.logger.error(f"Error uploading audio file: {str(e)}")
        import traceback
        current_app.logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to upload audio file: {str(e)}"}), 500

@transcribe_bp.route("/tasks/<task_id>/stream", methods=["GET"])
@token_required
def task_status_stream(user, task_id):
    """
    Server-Sent Events (SSE) endpoint that streams task status updates in real-time.
    Frontend can listen to this endpoint to get automatic updates without polling.
    
    Usage from frontend:
        const eventSource = new EventSource(`/tasks/${taskId}/stream`);
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            // Update UI with data
        };
    """
    def generate():
        """Generator function that yields SSE formatted messages"""
        try:
            last_state = None
            last_info = None
            check_interval = 0.5  # Check every 500ms
            max_duration = 3600  # Maximum 1 hour of streaming
            elapsed = 0
            
            while elapsed < max_duration:
                try:
                    result = AsyncResult(task_id, app=celery_app)
                    
                    # Prepare response data
                    if result.state == 'PENDING':
                        data = {
                            'id': task_id,
                            'state': result.state,
                            'current': 0,
                            'total': 100,
                            'status': 'Task is waiting to be processed...',
                            'progress_percent': 0
                        }
                    elif result.state == 'PROGRESS':
                        current = result.info.get('current', 0)
                        total = result.info.get('total', 100)
                        data = {
                            'id': task_id,
                            'state': result.state,
                            'current': current,
                            'total': total,
                            'status': result.info.get('status', 'Processing...'),
                            'progress_percent': int((current / total) * 100) if total > 0 else 0
                        }
                    elif result.state == 'SUCCESS':
                        data = {
                            'id': task_id,
                            'state': result.state,
                            'current': 100,
                            'total': 100,
                            'status': 'Task completed successfully',
                            'progress_percent': 100,
                            'result': result.result
                        }
                    elif result.state == 'FAILURE':
                        error_info = result.info if isinstance(result.info, dict) else {}
                        data = {
                            'id': task_id,
                            'state': result.state,
                            'current': 0,
                            'total': 100,
                            'status': 'Task failed',
                            'progress_percent': 0,
                            'error': str(error_info.get('error', str(result.info)))
                        }
                    elif result.state == 'REVOKED':
                        data = {
                            'id': task_id,
                            'state': result.state,
                            'current': 0,
                            'total': 100,
                            'status': 'Task was cancelled',
                            'progress_percent': 0
                        }
                    else:
                        data = {
                            'id': task_id,
                            'state': result.state,
                            'status': f'Unknown task state: {result.state}',
                            'progress_percent': 0
                        }
                    
                    # Only send update if state or info changed
                    current_state = (result.state, json.dumps(data, sort_keys=True))
                    if current_state != last_state:
                        yield f"data: {json.dumps(data)}\n\n"
                        last_state = current_state
                    
                    # If task is in terminal state, close the stream
                    if result.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
                        break
                    
                    import time
                    time.sleep(check_interval)
                    elapsed += check_interval
                    
                except Exception as e:
                    current_app.logger.error(f"Error in SSE stream: {str(e)}")
                    error_data = {
                        'id': task_id,
                        'state': 'ERROR',
                        'error': 'Failed to check task status',
                        'progress_percent': 0
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    break
            
            # Send final close message
            yield f"event: close\ndata: {json.dumps({'message': 'Stream closed'})}\n\n"
            
        except GeneratorExit:
            # Client disconnected
            pass
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable buffering in nginx
            'Connection': 'keep-alive'
        }
    )


@transcribe_bp.route("/tasks/<task_id>", methods=["GET"])
@token_required
def task_status(user, task_id):
    """Get the status of a background transcription task (polling endpoint)."""
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        if result.state == 'PENDING':
            response = {
                'id': task_id,
                'state': result.state,
                'current': 0,
                'total': 100,
                'status': 'Task is waiting to be processed...',
                'progress_percent': 0
            }
        elif result.state == 'PROGRESS':
            current = result.info.get('current', 0)
            total = result.info.get('total', 100)
            response = {
                'id': task_id,
                'state': result.state,
                'current': current,
                'total': total,
                'status': result.info.get('status', 'Processing...'),
                'progress_percent': int((current / total) * 100) if total > 0 else 0
            }
        elif result.state == 'SUCCESS':
            response = {
                'id': task_id,
                'state': result.state,
                'current': 100,
                'total': 100,
                'status': 'Task completed successfully',
                'progress_percent': 100,
                'result': result.result
            }
        elif result.state == 'FAILURE':
            error_info = result.info if isinstance(result.info, dict) else {}
            response = {
                'id': task_id,
                'state': result.state,
                'current': 0,
                'total': 100,
                'status': 'Task failed',
                'progress_percent': 0,
                'error': str(error_info.get('error', str(result.info)))
            }
        elif result.state == 'REVOKED':
            response = {
                'id': task_id,
                'state': result.state,
                'current': 0,
                'total': 100,
                'status': 'Task was cancelled',
                'progress_percent': 0
            }
        else:
            response = {
                'id': task_id,
                'state': result.state,
                'status': f'Unknown task state: {result.state}',
                'progress_percent': 0
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        current_app.logger.error(f"Error checking task status: {str(e)}")
        return jsonify({
            'id': task_id,
            'state': 'ERROR',
            'error': 'Failed to check task status',
            'progress_percent': 0
        }), 500


@transcribe_bp.route("/tasks/<task_id>/cancel", methods=["POST"])
@token_required
def cancel_task(user, task_id):
    """
    Cancel a running background task.
    Note: This will attempt to revoke the task. If the task is already running,
    it may not stop immediately depending on what it's currently doing.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        # Check if task is in a state that can be cancelled
        if result.state in ['SUCCESS', 'FAILURE', 'REVOKED']:
            return jsonify({
                'id': task_id,
                'state': result.state,
                'message': f'Task is already in terminal state: {result.state}',
                'cancelled': False
            }), 400
        
        # Revoke the task
        # terminate=True will kill the worker process (more aggressive)
        # terminate=False will just mark it as revoked (softer approach)
        result.revoke(terminate=False, signal='SIGTERM')
        
        current_app.logger.info(f"Task {task_id} cancellation requested")
        
        return jsonify({
            'id': task_id,
            'message': 'Task cancellation requested',
            'cancelled': True
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error cancelling task: {str(e)}")
        return jsonify({
            'id': task_id,
            'error': 'Failed to cancel task',
            'cancelled': False
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