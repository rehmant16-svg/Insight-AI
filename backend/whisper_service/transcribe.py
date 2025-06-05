import os
import whisper
import yt_dlp
import ffmpeg
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories for storing audio files
DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(exist_ok=True)

def download_youtube_audio(url, output_path):
    """
    Download audio from YouTube video
    """
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'outtmpl': str(output_path),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Downloading audio from {url}")
            ydl.download([url])
            
        return True
    except Exception as e:
        logger.error(f"Error downloading audio: {e}")
        return False

def transcribe_audio(audio_path):
    """
    Transcribe audio using Whisper
    """
    try:
        # Load Whisper model (using small model for faster processing)
        logger.info("Loading Whisper model...")
        model = whisper.load_model("small")
        
        # Transcribe the audio
        logger.info(f"Transcribing audio: {audio_path}")
        result = model.transcribe(str(audio_path))
        
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def process_youtube_video(url):
    """
    Main function to process YouTube video: download and transcribe
    """
    try:
        # Generate unique filename
        video_id = url.split("v=")[-1]
        output_path = DOWNLOAD_DIR / f"{video_id}.%(ext)s"
        
        # Download audio
        if not download_youtube_audio(url, output_path):
            return None
            
        # Get the actual WAV file path
        wav_path = output_path.with_suffix('.wav')
        
        # Transcribe the audio
        transcription = transcribe_audio(wav_path)
        
        # Clean up: remove the downloaded file
        try:
            os.remove(wav_path)
            logger.info(f"Cleaned up temporary file: {wav_path}")
        except Exception as e:
            logger.warning(f"Could not remove temporary file: {e}")
            
        return transcription
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return None

if __name__ == "__main__":
    # Test the transcription
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    result = process_youtube_video(test_url)
    if result:
        print("Transcription:", result)
    else:
        print("Transcription failed") 