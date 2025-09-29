# transcription_utils.py
import os
import gc
import re
import unicodedata
import subprocess
import tempfile
import uuid
import requests
import yt_dlp as youtube_dl
from urllib.parse import urlparse
from openai import OpenAI
from config import Config
from langchain.text_splitter import RecursiveCharacterTextSplitter
import imageio_ffmpeg
import logging

# Set up logging for Celery tasks
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def preprocess_and_chunk(text):
    """Chunk Text Using RecursiveCharacterTextSplitter"""
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
    """
    Note: This function requires Flask app context for embeddings and Pinecone.
    For Celery tasks, we'll just return the chunk count for now.
    You'll need to handle embeddings in a separate service or modify this function.
    """
    logger.info(f"Would process {len(chunks)} chunks for namespace {safe_namespace}")
    return len(chunks)  # Just return count for now

def transcribe_audio_with_openai(audio_path: str):
    """Send audio to OpenAI Whisper for transcription with retry logic."""
    max_retries = 3
    logger.info(f"Starting transcription for file: {audio_path}")
    
    # Check if OpenAI client is available
    if not client:
        logger.error("OpenAI client not initialized - check API key")
        return ""
    
    # Check if file exists and get size
    if not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist: {audio_path}")
        return ""
    
    file_size = os.path.getsize(audio_path)
    logger.info(f"Audio file size: {file_size} bytes")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Transcription attempt {attempt + 1}/{max_retries}")
            with open(audio_path, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",  # Use whisper-1 for better performance
                    file=f
                )
            
            if transcript and transcript.text:
                logger.info(f"Transcription successful, text length: {len(transcript.text)}")
                return transcript.text
            else:
                logger.warning(f"Transcription returned empty result for {audio_path}")
                return ""
                
        except Exception as e:
            error_msg = f"Transcription attempt {attempt + 1} failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Error type: {type(e).__name__}")
            
            if attempt == max_retries - 1:
                logger.error(f"All transcription attempts failed for {audio_path}")
                return ""
            import time
            time.sleep(2)  # Wait before retry
    
    return ""

def split_audio_into_chunks_ffmpeg(audio_path, output_dir="chunks", chunk_length=30):
    os.makedirs(output_dir, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # ✅ points to the bundled ffmpeg.exe
    try:
        subprocess.run([
            ffmpeg_exe, "-i", audio_path,
            "-f", "segment", "-segment_time", str(chunk_length),
            "-c", "copy", os.path.join(output_dir, "chunk_%03d.wav")
        ], check=True)
    except Exception as e:
        print(f"[ERROR] ffmpeg chunking failed: {e}")
        return []

    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")]

def transcribe_large_audio_optimized(audio_path: str):
    """Optimized large audio transcription with concurrent processing and memory management."""
    logger.info(f"Starting large audio transcription for: {audio_path}")
    
    # Check if input file exists
    if not os.path.exists(audio_path):
        logger.error(f"Input audio file does not exist: {audio_path}")
        return ""
    
    input_file_size = os.path.getsize(audio_path)
    logger.info(f"Input file size: {input_file_size} bytes")
    
    try:
        # Removed MAX_FILE_SIZE limit - let system handle large files naturally
        all_chunks = split_audio_into_chunks_ffmpeg(audio_path, chunk_length=60)

        if not all_chunks:
            logger.error("Failed to split audio into chunks")
            return ""
        
        transcripts = []
        total_chunks = len(all_chunks)
        logger.info(f"Successfully created {total_chunks} audio chunks")
        
        # Process chunks in batches to manage memory
        MAX_CONCURRENT_CHUNKS = 10
        for i in range(0, len(all_chunks), MAX_CONCURRENT_CHUNKS):
            batch_chunks = all_chunks[i:i + MAX_CONCURRENT_CHUNKS]
            batch_num = i // MAX_CONCURRENT_CHUNKS + 1
            total_batches = (total_chunks + MAX_CONCURRENT_CHUNKS - 1) // MAX_CONCURRENT_CHUNKS
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_chunks)} chunks")
            
            # Use ThreadPoolExecutor for concurrent processing
            from concurrent.futures import ThreadPoolExecutor, as_completed
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
                            logger.info(f"Successfully transcribed chunk: {chunk_path}")
                        else:
                            logger.warning(f"Empty transcription for chunk: {chunk_path}")
                    except Exception as e:
                        logger.error(f"Error transcribing chunk {chunk_path}: {str(e)}")
                    finally:
                        # Cleanup chunk file
                        try:
                            os.remove(chunk_path)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup chunk file {chunk_path}: {str(e)}")
            
            # Force garbage collection after each batch
            gc.collect()
        
        final_transcript = " ".join(transcripts)
        logger.info(f"Completed transcription of {total_chunks} chunks, final text length: {len(final_transcript)}")
        
        if not final_transcript.strip():
            logger.error("Final transcript is empty")
            return ""
            
        return final_transcript
        
    except Exception as e:
        logger.error(f"Error in transcribe_large_audio_optimized: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return ""

def find_stream_url_from_un(page_url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        # Removed timeout limit for production
        resp = requests.get(page_url, headers=headers, timeout=None)
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
        "hls_use_mpegts": True,
        # ✅ Force yt_dlp to use bundled ffmpeg from imageio
        "ffmpeg_location": os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()),
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(final_url, download=True)
            audio_path = ydl.prepare_filename(info)  # actual downloaded file path
        return (audio_path if os.path.exists(audio_path) else None), video_title
    except Exception as e:
        logger.error(f"yt_dlp download error: {e}")
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
            # Removed timeout limit for production
            timeout=None,
        )
        if os.path.exists(output_audio_path):
            try:
                os.remove(input_audio_path)
            except Exception:
                pass
            return output_audio_path
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        return None
