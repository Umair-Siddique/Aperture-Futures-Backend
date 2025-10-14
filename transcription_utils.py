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
import shutil
import psutil
import time

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

    ffmpeg_exe = get_ffmpeg_executable()
    if not ffmpeg_exe:
        logger.error("No FFmpeg executable found")
        return []
    
    if not test_ffmpeg_executable(ffmpeg_exe):
        logger.error(f"FFmpeg executable failed test: {ffmpeg_exe}")
        return []

    try:
        subprocess.run([
            ffmpeg_exe, "-i", audio_path,
            "-f", "segment", "-segment_time", str(chunk_length),
            "-c", "copy", os.path.join(output_dir, "chunk_%03d.wav")
        ], check=True)
    except Exception as e:
        logger.error(f"FFmpeg chunking failed: {e}")
        return []

    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".wav")]

def transcribe_large_audio_optimized(audio_path: str):
    """Optimized large audio transcription with sequential processing and memory management."""
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
        
        # Process chunks sequentially to avoid Flask context issues
        logger.info("Processing chunks sequentially to avoid context issues")
        
        for i, chunk_path in enumerate(all_chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}: {chunk_path}")
            
            try:
                text = transcribe_audio_with_openai(chunk_path)
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
                    logger.info(f"Cleaned up chunk: {chunk_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup chunk file {chunk_path}: {str(e)}")
            
            # Check memory usage periodically
            if (i + 1) % 5 == 0:  # Every 5 chunks
                check_memory_usage()
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

    ffmpeg_exe = get_ffmpeg_executable()
    if not ffmpeg_exe:
        logger.error("No FFmpeg executable found for yt-dlp")
        return None, None
    
    ydl_opts = {
        # Download the best available format (audio or video) - we'll extract audio manually
        "format": "best",
        "nocheckcertificate": True,
        "quiet": False,  # Enable output to see what's happening
        "outtmpl": outtmpl,
        "hls_use_mpegts": True,
        # Use the best available FFmpeg for downloading only
        "ffmpeg_location": os.path.dirname(ffmpeg_exe),
        # Explicitly disable all post-processing
        "postprocessors": [],
        "embed_subs": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "writedescription": False,
        "writeinfojson": False,
        "writethumbnail": False,
    }

    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Starting download from: {final_url}")
            info = ydl.extract_info(final_url, download=True)
            
            # Get the actual downloaded file path
            if 'requested_downloads' in info and info['requested_downloads']:
                # Use the actual downloaded file path
                audio_path = info['requested_downloads'][0]['filepath']
            else:
                # Fallback to prepare_filename
                audio_path = ydl.prepare_filename(info)
            
            logger.info(f"Downloaded file: {audio_path}")
            
            # Verify file exists
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path)
                logger.info(f"Downloaded file size: {file_size} bytes")
                return audio_path, video_title
            else:
                logger.error(f"Downloaded file not found: {audio_path}")
                return None, None
                
    except Exception as e:
        logger.error(f"yt_dlp download error: {e}")
        return None, None

# FFmpeg utility functions
def get_ffmpeg_executable():
    """Get the best available FFmpeg executable (system first, then bundled)."""
    # Try system FFmpeg first (usually has better codec support)
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        logger.info(f"Using system FFmpeg: {system_ffmpeg}")
        return system_ffmpeg
    
    # Fallback to bundled FFmpeg
    try:
        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        logger.info(f"Using bundled FFmpeg: {bundled_ffmpeg}")
        return bundled_ffmpeg
    except Exception as e:
        logger.error(f"Failed to get bundled FFmpeg: {e}")
        return None

def test_ffmpeg_executable(ffmpeg_path):
    """Test if FFmpeg executable works and has required codecs."""
    if not ffmpeg_path:
        return False
    
    try:
        # Test basic FFmpeg functionality
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info(f"FFmpeg test successful: {ffmpeg_path}")
            # Check for libmp3lame codec
            if "libmp3lame" in result.stdout:
                logger.info("libmp3lame codec available")
                return True
            else:
                logger.warning("libmp3lame codec not found")
                return True  # Still usable, just without mp3lame
        else:
            logger.error(f"FFmpeg test failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"FFmpeg test error: {e}")
        return False

def check_memory_usage():
    """Check current memory usage and trigger cleanup if needed."""
    memory_percent = psutil.virtual_memory().percent
    # Removed memory threshold check - let system handle memory naturally
    logger.info(f"Current memory usage: {memory_percent}%")
    gc.collect()
    return False

def convert_and_compress_audio_optimized(input_audio_path: str):
    """Optimized audio conversion with better compression and memory management."""
    if not input_audio_path or not os.path.exists(input_audio_path):
        logger.error(f"Input audio file does not exist: {input_audio_path}")
        return None
    
    output_audio_path = os.path.splitext(input_audio_path)[0] + ".mp3"
    ffmpeg_exe = get_ffmpeg_executable()
    
    if not ffmpeg_exe:
        logger.error("No FFmpeg executable found")
        return None
    
    logger.info(f"Converting audio: {input_audio_path} -> {output_audio_path}")
    logger.info(f"Using FFmpeg: {ffmpeg_exe}")
    
    # Test FFmpeg before using it
    if not test_ffmpeg_executable(ffmpeg_exe):
        logger.error(f"FFmpeg executable failed test: {ffmpeg_exe}")
        return None
    
    # Analyze input file streams to check for audio
    logger.info("Analyzing input file streams...")
    try:
        probe_result = subprocess.run(
            [ffmpeg_exe, "-i", input_audio_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        # FFmpeg writes stream info to stderr
        stream_info = probe_result.stderr
        logger.info(f"Stream info: {stream_info}")
        
        # Check if there are audio streams - be more specific
        has_audio = False
        has_video = False
        audio_streams = []
        video_streams = []
        
        # Parse stream information more carefully
        for line in stream_info.split('\n'):
            if 'Stream #' in line:
                if 'Audio:' in line:
                    has_audio = True
                    audio_streams.append(line.strip())
                elif 'Video:' in line:
                    has_video = True
                    video_streams.append(line.strip())
        
        logger.info(f"File analysis - Has audio: {has_audio}, Has video: {has_video}")
        if audio_streams:
            logger.info(f"Audio streams found: {audio_streams}")
        if video_streams:
            logger.info(f"Video streams found: {video_streams}")
        
        # If no audio streams found, return immediately with specific error
        if not has_audio:
            logger.error("No audio streams found in the input file. This appears to be a video-only file without audio content.")
            return None
        
    except Exception as e:
        logger.warning(f"Failed to analyze streams: {e}")
        # Continue with conversion attempts even if analysis fails
    
    # Try multiple conversion strategies
    conversion_strategies = []
    
    # Add video-specific strategies first (extract audio from video)
    # Put most compatible strategies first
    conversion_strategies.extend([
        # Strategy 1: Extract audio with basic mp3 encoder (most compatible)
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-vn",  # No video
            "-acodec", "mp3",
            "-ab", "128k",
            output_audio_path,
        ],
        # Strategy 2: Extract first audio stream with basic mp3
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-map", "0:a:0",  # Map first audio stream
            "-acodec", "mp3",
            "-ab", "64k",
            output_audio_path,
        ],
        # Strategy 3: Extract audio from video with libmp3lame (if available)
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ab", "64k",
            "-ar", "22050",
            "-ac", "1",
            output_audio_path,
        ],
        # Strategy 4: Try to extract any audio stream with libmp3lame
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-map", "0:a?",  # Map audio streams if they exist
            "-acodec", "libmp3lame",
            "-ab", "64k",
            output_audio_path,
        ],
    ])
    
    # Add audio file strategies (for files that are already audio)
    # Put most compatible strategies first
    conversion_strategies.extend([
        # Strategy 5: Basic conversion without advanced options (most compatible)
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-acodec", "mp3",
            output_audio_path,
        ],
        # Strategy 6: Copy audio stream if it's already compatible
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-c:a", "copy",
            "-f", "mp3",
            output_audio_path,
        ],
        # Strategy 7: Try libmp3lame if available
        [
            ffmpeg_exe,
            "-y",
            "-i", input_audio_path,
            "-acodec", "libmp3lame",
            "-ab", "64k",
            "-ar", "22050",
            "-ac", "1",
            output_audio_path,
        ],
        # Strategy 8: Standard conversion with libmp3lame (last resort)
        [
            ffmpeg_exe,
            "-y",  # Overwrite output files
            "-i", input_audio_path,
            "-acodec", "libmp3lame",
            "-ab", "32k",
            "-ar", "16000",
            "-ac", "1",
            "-af", "volume=0.8",
            output_audio_path,
        ]
    ])
    
    for strategy_num, cmd in enumerate(conversion_strategies, 1):
        logger.info(f"Trying conversion strategy {strategy_num}: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per strategy
            )
            
            logger.info(f"Strategy {strategy_num} - FFmpeg stdout: {result.stdout}")
            if result.stderr:
                logger.info(f"Strategy {strategy_num} - FFmpeg stderr: {result.stderr}")
            
            if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
                logger.info(f"Audio conversion successful with strategy {strategy_num}: {output_audio_path}")
                try:
                    os.remove(input_audio_path)
                    logger.info(f"Removed original file: {input_audio_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove original file: {e}")
                return output_audio_path
            else:
                logger.warning(f"Strategy {strategy_num} did not create valid output file")
                # Clean up empty or invalid output file
                try:
                    if os.path.exists(output_audio_path):
                        os.remove(output_audio_path)
                except Exception:
                    pass
                    
        except subprocess.CalledProcessError as e:
            logger.warning(f"Strategy {strategy_num} failed with exit code {e.returncode}")
            logger.warning(f"Strategy {strategy_num} - FFmpeg stdout: {e.stdout}")
            logger.warning(f"Strategy {strategy_num} - FFmpeg stderr: {e.stderr}")
            continue
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Strategy {strategy_num} timed out: {str(e)}")
            continue
        except Exception as e:
            logger.warning(f"Strategy {strategy_num} unexpected error: {str(e)}")
            continue
    
    # If all strategies failed
    logger.error("All conversion strategies failed")
    return None
