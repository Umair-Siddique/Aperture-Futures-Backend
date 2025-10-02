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
def transcribe_audio_task(self, title, description, members_list, file_path):
    """
    Background task to transcribe uploaded audio with progress reporting.
    """
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting transcription...'}
        )
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {'ok': False, 'error': f'Audio file not found: {file_path}'}
        
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Transcribing audio...'}
        )
        
        # Try large audio transcription first
        transcript = transcribe_large_audio_optimized(file_path)
        if not transcript:
            # Fallback to OpenAI transcription
            self.update_state(
                state='PROGRESS',
                meta={'current': 30, 'total': 100, 'status': 'Trying alternative transcription method...'}
            )
            transcript = transcribe_audio_with_openai(file_path)
        
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
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
            print(f"Cleaned up file: {file_path}")
        except Exception as e:
            print(f"Failed to cleanup file {file_path}: {e}")
        
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
        
        # Clean up uploaded file on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        
        return {'ok': False, 'error': error_msg}


@celery_app.task(bind=True, name="transcribe.video")
def transcribe_video_task(self, title, description, members_list, video_url):
    """
    Background task to download audio from video and transcribe with progress reporting.
    """
    try:
        # Update task state to PROGRESS
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting video processing...'}
        )
        
        # Download audio
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Downloading audio from video...'}
        )
        
        audio_path, _ = download_audio(video_url)
        if not audio_path:
            return {'ok': False, 'error': 'Audio download failed'}
        
        # Convert and compress
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Converting and compressing audio...'}
        )
        
        tmp_mp3 = convert_and_compress_audio_optimized(audio_path)
        if not tmp_mp3:
            return {'ok': False, 'error': 'Audio conversion/compression failed'}
        
        # Transcribe
        self.update_state(
            state='PROGRESS',
            meta={'current': 50, 'total': 100, 'status': 'Transcribing audio...'}
        )
        
        transcript = transcribe_large_audio_optimized(tmp_mp3)
        if not transcript:
            return {'ok': False, 'error': 'Transcription failed'}
        
        # Generate report with Flask app context
        self.update_state(
            state='PROGRESS',
            meta={'current': 80, 'total': 100, 'status': 'Generating report...'}
        )
        
        report_saved = False
        try:
            with create_app_context():
                from report.generate_report import generate_and_store_transcription_report
                report_info = generate_and_store_transcription_report(title=title, transcript=transcript)
                report_saved = bool(report_info.get("ok"))
        except Exception as e:
            print(f"Report generation failed: {e}")
        
        # Process embeddings with Flask app context
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Processing embeddings...'}
        )
        
        chunks_stored = 0
        try:
            with create_app_context():
                from flask import current_app
                chunks = preprocess_and_chunk(transcript)
                safe_namespace = sanitize_id(title)
                
                # Process embeddings (same logic as audio task)
                total_chunks = len(chunks)
                batch_size = 8
                
                for i in range(0, total_chunks, batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    
                    try:
                        batch_embeds = current_app.embeddings.embed_documents(batch_chunks)
                        vectors = [
                            (f"{safe_namespace}_{i+j}", vec, {"text": chunk})
                            for j, (vec, chunk) in enumerate(zip(batch_embeds, batch_chunks))
                        ]
                        current_app.pinecone_index.upsert(vectors=vectors, namespace=safe_namespace)
                        chunks_stored += len(vectors)
                        del batch_embeds, vectors
                        gc.collect()
                    except Exception as e:
                        print(f"Error processing embedding batch: {str(e)}")
                        continue
                
        except Exception as e:
            print(f"Embedding processing failed: {e}")
        
        # Store in Supabase with Flask app context
        supabase_stored = False
        try:
            with create_app_context():
                from flask import current_app
                import time
                
                current_app.supabase.table("audio_files").insert({
                    "title": title,
                    "description": description,
                    "members": members_list,
                    "timestamp": int(time.time()),
                }).execute()
                supabase_stored = True
        except Exception as e:
            print(f"Failed to store in Supabase: {e}")
        
        # Cleanup
        del transcript
        if 'chunks' in locals():
            del chunks
        gc.collect()
        
        # Clean up temporary files
        try:
            if tmp_mp3 and os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except Exception:
            pass
        
        return {
            "ok": True,
            "title": title,
            "description": description,
            "members": members_list,
            "url": video_url,
            "chunks_stored": chunks_stored,
            "report_saved": report_saved,
            "supabase_stored": supabase_stored,
        }
        
    except Exception as e:
        error_msg = str(e)
        return {'ok': False, 'error': error_msg}
