# tasks/transcribe_tasks.py
import os
import sys
import gc
from openai import OpenAI
from config import Config

# Add the current directory to Python path so we can import app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from transcription_utils import (
    transcribe_large_audio_optimized,
    transcribe_audio_with_openai,
    preprocess_and_chunk,
    sanitize_id,
    download_audio,
    convert_and_compress_audio_optimized
)

# Initialize clients here (not via current_app)
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def create_app_context():
    """Create Flask application context for Celery tasks"""
    try:
        # Import here to avoid circular imports
        import sys
        import os
        
        # Ensure we can import from the parent directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Now import the app
        from app import create_app
        app = create_app()
        return app.app_context()
    except Exception as e:
        print(f"Error creating app context: {e}")
        import traceback
        print(f"App context error traceback: {traceback.format_exc()}")
        raise

# Import celery_app here to avoid circular imports
# This will be imported when the task is actually called
def get_celery_app():
    from celery_app import celery_app
    return celery_app

# Get the celery app instance
celery_app = get_celery_app()

@celery_app.task(bind=True, name="transcribe.audio")
def transcribe_audio_task(self, title, description, members_list, storage_url, filename):
    """
    Background task to transcribe uploaded audio with progress reporting.
    """
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting transcription...'}
        )
        
        # Download file from Supabase storage to local temp file
        self.update_state(
            state='PROGRESS',
            meta={'current': 5, 'total': 100, 'status': 'Downloading audio file...'}
        )
        
        local_file_path = None
        try:
            with create_app_context():
                from flask import current_app
                import requests
                import tempfile
                
                # Download the file from Supabase storage
                response = requests.get(storage_url)
                response.raise_for_status()
                
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                local_file_path = os.path.join(temp_dir, filename)
                
                with open(local_file_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded audio file from storage to: {local_file_path}")
                
        except Exception as e:
            print(f"Failed to download audio file from storage: {e}")
            return {'ok': False, 'error': f'Failed to download audio file: {str(e)}'}
        
        # Check if file exists
        if not os.path.exists(local_file_path):
            return {'ok': False, 'error': f'Audio file not found after download: {local_file_path}'}
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Transcribing audio...'}
        )
        
        # Try large audio transcription first
        transcript = transcribe_large_audio_optimized(local_file_path)
        if not transcript:
            # Fallback to OpenAI transcription
            self.update_state(
                state='PROGRESS',
                meta={'current': 30, 'total': 100, 'status': 'Trying alternative transcription method...'}
            )
            transcript = transcribe_audio_with_openai(local_file_path)
        
        if not transcript:
            return {'ok': False, 'error': 'Transcription failed with all methods'}
        
        print(f"Transcription completed successfully, length: {len(transcript)}")
        
        # Store transcription in Supabase FIRST (before report generation)
        supabase_stored = False
        try:
            print("Starting Supabase storage...")
            with create_app_context():
                from flask import current_app
                import time
                
                result = current_app.supabase.table("audio_files").insert({
                    "title": title,
                    "description": description,
                    "members": members_list,
                    "timestamp": int(time.time()),
                }).execute()
                supabase_stored = True
                print(f"Supabase storage successful: {bool(result.data)}")
        except Exception as e:
            print(f"Failed to store in Supabase: {e}")
            import traceback
            print(f"Supabase error traceback: {traceback.format_exc()}")
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 60, 'total': 100, 'status': 'Generating report...'}
        )
        
        # Generate report with Flask app context (AFTER Supabase record exists)
        report_saved = False
        try:
            print("Starting report generation...")
            with create_app_context():
                from report.generate_report import generate_and_store_transcription_report
                report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
                report_saved = bool(report_info.get("ok"))
                print(f"Report generation result: {report_saved}")
        except Exception as e:
            print(f"Report generation failed: {e}")
            import traceback
            print(f"Report error traceback: {traceback.format_exc()}")

        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Processing embeddings...'}
        )
        
        # Process embeddings with Flask app context
        chunks_stored = 0
        try:
            print("Starting embedding processing...")
            with create_app_context():
                from flask import current_app
                chunks = preprocess_and_chunk(transcript)
                safe_namespace = sanitize_id(title)
                print(f"Created {len(chunks)} chunks for namespace: {safe_namespace}")
                
                # Process embeddings in batches
                total_chunks = len(chunks)
                batch_size = 8
                
                for i in range(0, total_chunks, batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1} with {len(batch_chunks)} chunks")
                    
                    try:
                        # Generate embeddings for this batch
                        batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
                        print(f"Generated {len(batch_embeds)} embeddings")
                        
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
                        chunks_stored += len(vectors)
                        print(f"Stored {len(vectors)} vectors in Pinecone")
                        
                        # Clear batch data from memory
                        del batch_embeds, vectors
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error processing embedding batch {i//batch_size + 1}: {str(e)}")
                        import traceback
                        print(f"Embedding batch error traceback: {traceback.format_exc()}")
                        continue
                
                print(f"Total chunks stored: {chunks_stored}")
                
        except Exception as e:
            print(f"Embedding processing failed: {e}")
            import traceback
            print(f"Embedding error traceback: {traceback.format_exc()}")
        
        # Cleanup
        del transcript
        if 'chunks' in locals():
            del chunks
        gc.collect()
        
        # Clean up local temporary file
        try:
            if local_file_path and os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Cleaned up local file: {local_file_path}")
        except Exception as e:
            print(f"Failed to cleanup local file {local_file_path}: {e}")
        
        # Clean up file from Supabase storage
        try:
            with create_app_context():
                from flask import current_app
                current_app.supabase.storage.from_("audio_bucket").remove([filename])
                print(f"Cleaned up storage file: {filename}")
        except Exception as e:
            print(f"Failed to cleanup storage file {filename}: {e}")
        
        # Final result - DON'T call update_state after this
        final_result = {
            "ok": True,
            "title": title,
            "description": description,
            "members": members_list,
            "chunks_stored": chunks_stored,
            "report_saved": report_saved,
            "supabase_stored": supabase_stored,
        }
        
        print(f"Task completed with result: {final_result}")
        return final_result
        
    except Exception as e:
        # Simple error handling
        error_msg = str(e)
        print(f"Task failed with error: {error_msg}")
        import traceback
        print(f"Task error traceback: {traceback.format_exc()}")
        
        # Clean up local file on error
        try:
            if 'local_file_path' in locals() and local_file_path and os.path.exists(local_file_path):
                os.remove(local_file_path)
        except Exception:
            pass
        
        # Clean up storage file on error
        try:
            with create_app_context():
                from flask import current_app
                current_app.supabase.storage.from_("audio_bucket").remove([filename])
        except Exception:
            pass
        
        return {'ok': False, 'error': error_msg}


@celery_app.task(bind=True, name="transcribe.video")
def transcribe_video_task(self, title, description, members_list, video_url):
    """
    Background task to download audio from video and transcribe with progress reporting.
    """
    audio_path = None
    tmp_mp3 = None
    
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting video processing...'}
        )
        
        # Check memory usage before starting
        from transcription_utils import check_memory_usage
        check_memory_usage()
        
        # Download audio
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Downloading audio from video...'}
        )
        
        audio_path, inferred_title = download_audio(video_url)
        if not audio_path:
            return {'ok': False, 'error': 'Audio download failed'}
        
        print(f"Downloaded audio from video: {audio_path}")
        
        # Convert and compress audio
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Converting and compressing audio...'}
        )
        
        tmp_mp3 = convert_and_compress_audio_optimized(audio_path)
        if not tmp_mp3:
            print("Audio conversion failed, trying to use original file directly")
            # Try to use the original file directly if it's audio
            if audio_path and os.path.exists(audio_path):
                file_extension = os.path.splitext(audio_path)[1].lower()
                if file_extension in ['.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac']:
                    print(f"Using original audio file directly: {audio_path}")
                    tmp_mp3 = audio_path
                else:
                    print("Video file appears to have no audio streams or unsupported format")
                    return {
                        'ok': False, 
                        'error': 'The provided video file does not contain any audio streams. This appears to be a video-only file without audio content. Please provide a video file that includes audio for transcription.'
                    }
            else:
                return {'ok': False, 'error': 'Audio download or conversion failed'}
        
        # Clean up original audio file (only if we converted it)
        try:
            if audio_path and os.path.exists(audio_path) and audio_path != tmp_mp3:
                os.remove(audio_path)
                print(f"Cleaned up original audio file: {audio_path}")
        except Exception as e:
            print(f"Failed to clean up original audio file: {e}")
        
        # Transcribe
        self.update_state(
            state='PROGRESS',
            meta={'current': 50, 'total': 100, 'status': 'Transcribing audio...'}
        )
        
        transcript = transcribe_large_audio_optimized(tmp_mp3)
        if not transcript:
            return {'ok': False, 'error': 'Transcription failed'}
        
        print(f"Transcription completed successfully, length: {len(transcript)}")
        
        # Store transcription in Supabase FIRST (before report generation)
        supabase_stored = False
        try:
            print("Starting Supabase storage...")
            with create_app_context():
                from flask import current_app
                import time
                
                result = current_app.supabase.table("audio_files").insert({
                    "title": title,
                    "description": description,
                    "members": members_list,
                    "timestamp": int(time.time()),
                }).execute()
                supabase_stored = True
                print(f"Supabase storage successful: {bool(result.data)}")
                
                # Add a small delay to ensure record is fully committed
                time.sleep(1)
                
        except Exception as e:
            print(f"Failed to store in Supabase: {e}")
            import traceback
            print(f"Supabase error traceback: {traceback.format_exc()}")
        
        # Generate report with Flask app context (AFTER Supabase record exists)
        self.update_state(
            state='PROGRESS',
            meta={'current': 70, 'total': 100, 'status': 'Generating report...'}
        )
        
        report_saved = False
        try:
            print("Starting report generation...")
            print(f"Supabase storage status: {supabase_stored}")
            print(f"Title for report: '{title}'")
            print(f"Transcript length: {len(transcript)} characters")
            
            # Only generate report if Supabase storage was successful
            if supabase_stored:
                with create_app_context():
                    from report.generate_report import generate_and_store_transcription_report
                    from flask import current_app
                    
                    # Verify the record exists before generating report
                    try:
                        existing_record = current_app.supabase.table("audio_files").select("title").eq("title", title).execute()
                        print(f"Record verification - found records: {len(existing_record.data) if existing_record.data else 0}")
                        
                        if existing_record.data:
                            print("Record confirmed, proceeding with report generation...")
                            report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
                            report_saved = bool(report_info.get("ok"))
                            print(f"Report generation result: {report_saved}")
                            print(f"Report info: {report_info}")
                        else:
                            print("ERROR: Record not found in database, cannot generate report")
                            
                    except Exception as verify_error:
                        print(f"Error verifying record existence: {verify_error}")
                        # Still try to generate report
                        report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
                        report_saved = bool(report_info.get("ok"))
                        print(f"Report generation result (after verification error): {report_saved}")
                        
            else:
                print("Skipping report generation - Supabase record not created")
        except Exception as e:
            print(f"Report generation failed: {e}")
            import traceback
            print(f"Report error traceback: {traceback.format_exc()}")
        
        # Process embeddings with Flask app context
        self.update_state(
            state='PROGRESS',
            meta={'current': 85, 'total': 100, 'status': 'Processing embeddings...'}
        )
        
        chunks_stored = 0
        try:
            print("Starting embedding processing...")
            with create_app_context():
                from flask import current_app
                chunks = preprocess_and_chunk(transcript)
                safe_namespace = sanitize_id(title)
                print(f"Created {len(chunks)} chunks for namespace: {safe_namespace}")
                
                # Process embeddings in batches
                total_chunks = len(chunks)
                batch_size = 8
                
                for i in range(0, total_chunks, batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    print(f"Processing batch {i//batch_size + 1} with {len(batch_chunks)} chunks")
                    
                    try:
                        # Generate embeddings for this batch
                        batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
                        print(f"Generated {len(batch_embeds)} embeddings")
                        
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
                        chunks_stored += len(vectors)
                        print(f"Stored {len(vectors)} vectors in Pinecone")
                        
                        # Clear batch data from memory
                        del batch_embeds, vectors
                        gc.collect()
                        
                    except Exception as e:
                        print(f"Error processing embedding batch {i//batch_size + 1}: {str(e)}")
                        import traceback
                        print(f"Embedding batch error traceback: {traceback.format_exc()}")
                        continue
                
                print(f"Total chunks stored: {chunks_stored}")
                
        except Exception as e:
            print(f"Embedding processing failed: {e}")
            import traceback
            print(f"Embedding error traceback: {traceback.format_exc()}")
        
        # Cleanup
        del transcript
        if 'chunks' in locals():
            del chunks
        gc.collect()
        
        # Clean up temporary files
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
                print(f"Cleaned up temporary audio file: {tmp_mp3}")
        except Exception as e:
            print(f"Failed to clean up temporary audio file: {e}")
        
        try:
            if audio_path and os.path.exists(audio_path) and audio_path != tmp_mp3:
                os.remove(audio_path)
                print(f"Cleaned up original audio file: {audio_path}")
        except Exception as e:
            print(f"Failed to clean up original audio file: {e}")
        
        # Final result - DON'T call update_state after this
        final_result = {
            "ok": True,
            "title": title,
            "description": description,
            "members": members_list,
            "url": video_url,
            "chunks_stored": chunks_stored,
            "report_saved": report_saved,
            "supabase_stored": supabase_stored,
        }
        
        print(f"Task completed with result: {final_result}")
        return final_result
        
    except Exception as e:
        # Simple error handling
        error_msg = str(e)
        print(f"Task failed with error: {error_msg}")
        import traceback
        print(f"Task error traceback: {traceback.format_exc()}")
        
        # Clean up temporary files on error
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass
        
        try:
            if audio_path and os.path.exists(audio_path) and audio_path != tmp_mp3:
                os.remove(audio_path)
        except Exception:
            pass
        
        return {'ok': False, 'error': error_msg}


@celery_app.task(bind=True, name="process.resolution")
def process_resolution_task(self, filename, storage_url):
    """
    Background task to process uploaded PDF resolution with progress reporting.
    """
    local_file_path = None
    
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting PDF processing...'}
        )
        
        # Download file from Supabase storage to local temp file
        self.update_state(
            state='PROGRESS',
            meta={'current': 5, 'total': 100, 'status': 'Downloading PDF file...'}
        )
        
        try:
            with create_app_context():
                from flask import current_app
                import requests
                import tempfile
                
                # Download the file from Supabase storage
                response = requests.get(storage_url)
                response.raise_for_status()
                
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                local_file_path = os.path.join(temp_dir, filename)
                
                with open(local_file_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded PDF file from storage to: {local_file_path}")
                
        except Exception as e:
            print(f"Failed to download PDF file from storage: {e}")
            return {'ok': False, 'error': f'Failed to download PDF file: {str(e)}'}
        
        # Check if file exists
        if not os.path.exists(local_file_path):
            return {'ok': False, 'error': f'PDF file not found after download: {local_file_path}'}
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Processing PDF...'}
        )
        
        # Process the PDF with the original logic
        try:
            with create_app_context():
                from flask import current_app
                from bluelines_backend.blueprints.upload_resolution import process_pdf_with_openai
                
                # Open the file and pass to processing function
                with open(local_file_path, 'rb') as pdf_file:
                    # Create a file-like object that process_pdf_with_openai expects
                    from io import BytesIO
                    file_content = pdf_file.read()
                    pdf_file_obj = BytesIO(file_content)
                    pdf_file_obj.seek(0)
                    
                    # Get the app instance
                    app = current_app._get_current_object()
                    
                    # Process the PDF
                    self.update_state(
                        state='PROGRESS',
                        meta={'current': 30, 'total': 100, 'status': 'Extracting text and metadata...'}
                    )
                    
                    result = process_pdf_with_openai(pdf_file_obj, filename, app)
                    
                    print(f"Successfully processed PDF: {filename}")
                    print(f"Result: {result}")
                    
        except Exception as e:
            print(f"Failed to process PDF: {e}")
            import traceback
            print(f"PDF processing error traceback: {traceback.format_exc()}")
            return {'ok': False, 'error': f'Failed to process PDF: {str(e)}'}
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Finalizing...'}
        )
        
        # Clean up local temporary file
        try:
            if local_file_path and os.path.exists(local_file_path):
                os.remove(local_file_path)
                print(f"Cleaned up local file: {local_file_path}")
        except Exception as e:
            print(f"Failed to cleanup local file {local_file_path}: {e}")
        
        # Clean up file from Supabase storage
        try:
            with create_app_context():
                from flask import current_app
                current_app.supabase.storage.from_("audio_bucket").remove([filename])
                print(f"Cleaned up storage file: {filename}")
        except Exception as e:
            print(f"Failed to cleanup storage file {filename}: {e}")
        
        # Final result
        final_result = {
            "ok": True,
            "message": "PDF processed and stored successfully",
            "data": result
        }
        
        print(f"Task completed with result: {final_result}")
        return final_result
        
    except Exception as e:
        # Simple error handling
        error_msg = str(e)
        print(f"Task failed with error: {error_msg}")
        import traceback
        print(f"Task error traceback: {traceback.format_exc()}")
        
        # Clean up local file on error
        try:
            if local_file_path and os.path.exists(local_file_path):
                os.remove(local_file_path)
        except Exception:
            pass
        
        # Clean up storage file on error
        try:
            with create_app_context():
                from flask import current_app
                current_app.supabase.storage.from_("audio_bucket").remove([filename])
        except Exception:
            pass
        
        return {'ok': False, 'error': error_msg}