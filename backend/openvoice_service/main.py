from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import os
import shutil
from pathlib import Path
import torch
import torchaudio
import numpy as np
from typing import Optional
import json
import sys
import subprocess
import logging
import librosa
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request models
class SpeechRequest(BaseModel):
    voice_id: str
    text: str

# Add OpenVoice to Python path
OPENVOICE_DIR = Path("OpenVoice")
sys.path.append(str(OPENVOICE_DIR))

# Create directories for storing audio files and models
UPLOAD_DIR = Path("uploads")
MODEL_DIR = Path("models")
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Initialize OpenVoice models
base_speaker_tts = None
tone_color_converter = None
try:
    logger.info("Attempting to import OpenVoice...")
    from openvoice.api import BaseSpeakerTTS, ToneColorConverter
    
    logger.info("Checking for model checkpoints...")
    # Update checkpoint paths to look in the correct locations
    ckpt_base = OPENVOICE_DIR / "checkpoints" / "base_speakers" / "EN"
    ckpt_converter = OPENVOICE_DIR / "checkpoints" / "converter"
    
    if not ckpt_base.exists() or not any(ckpt_base.iterdir()):
        logger.error("No base speaker checkpoints found in: " + str(ckpt_base))
    if not ckpt_converter.exists() or not any(ckpt_converter.iterdir()):
        logger.error("No converter checkpoints found in: " + str(ckpt_converter))
    
    if (ckpt_base.exists() and any(ckpt_base.iterdir())) and \
       (ckpt_converter.exists() and any(ckpt_converter.iterdir())):
        logger.info("Found model checkpoints. Initializing OpenVoice models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize base speaker TTS
        base_speaker_tts = BaseSpeakerTTS(
            config_path=str(ckpt_base / "config.json"),
            device=device
        )
        base_speaker_tts.load_ckpt(str(ckpt_base / "checkpoint.pth"))
        
        # Initialize tone color converter
        tone_color_converter = ToneColorConverter(
            config_path=str(ckpt_converter / "config.json"),
            device=device
        )
        tone_color_converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))
        
        logger.info("OpenVoice models initialized successfully!")
    else:
        logger.error("No valid checkpoints found in either base speakers or converter directories")
except ImportError as e:
    logger.error(f"Failed to import OpenVoice: {e}")
    logger.info("Please make sure OpenVoice is installed correctly:")
    logger.info("1. cd OpenVoice")
    logger.info("2. pip install -e .")
except Exception as e:
    logger.error(f"Failed to initialize OpenVoice models: {e}")
    logger.error("Please check if all dependencies are installed and model checkpoints are present.")

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/clone-voice")
async def clone_voice(
    audio: UploadFile = File(...),
    name: str = None
):
    try:
        if tone_color_converter is None:
            raise HTTPException(
                status_code=500, 
                detail="OpenVoice models not initialized. Please check the server logs for details."
            )

        # Validate file type
        if not audio.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
            raise HTTPException(
                status_code=400,
                detail="Only WAV, MP3, and OGG files are supported"
            )

        # Generate a unique name if none provided
        if name is None:
            name = f"voice_{os.urandom(4).hex()}"

        # Save the uploaded audio file
        file_path = UPLOAD_DIR / f"{name}_{audio.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        logger.info(f"Processing audio file: {file_path}")
        
        # Convert to WAV if needed
        if not file_path.suffix.lower() == '.wav':
            wav_path = file_path.with_suffix('.wav')
            logger.info(f"Converting {file_path} to WAV format...")
            try:
                # Load audio with librosa
                audio_data, sr = librosa.load(str(file_path), sr=16000)
                # Save as WAV
                import soundfile as sf
                sf.write(str(wav_path), audio_data, sr)
                file_path = wav_path
                logger.info(f"Successfully converted to WAV: {wav_path}")
            except Exception as e:
                logger.error(f"Error converting audio to WAV: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error converting audio to WAV: {str(e)}"
                )
        
        # Extract speaker embedding using the converter model
        try:
            from openvoice import se_extractor
            target_se, audio_name = se_extractor.get_se(
                str(file_path),
                tone_color_converter,
                target_dir=str(UPLOAD_DIR),
                vad=True
            )
            logger.info(f"Successfully extracted speaker embedding for {audio_name}")
        except Exception as e:
            logger.error(f"Error extracting speaker embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error extracting speaker embedding: {str(e)}"
            )
        
        # Save the speaker embedding
        embedding_path = UPLOAD_DIR / f"{name}_embedding.pt"
        try:
            torch.save(target_se, embedding_path)
            logger.info(f"Successfully saved speaker embedding: {embedding_path}")
        except Exception as e:
            logger.error(f"Error saving speaker embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saving speaker embedding: {str(e)}"
            )
        
        return {
            "status": "success",
            "message": "Voice embedding extracted successfully",
            "voiceId": name,
            "embeddingPath": str(embedding_path)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in clone_voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    src_path = None
    output_path = None
    try:
        if base_speaker_tts is None or tone_color_converter is None:
            raise HTTPException(
                status_code=500, 
                detail="OpenVoice models not initialized. Please check the server logs for details."
            )

        logger.info(f"Received request with voice_id: {request.voice_id}")
        logger.info(f"Text length: {len(request.text)}")
        
        # Check if this is a custom voice (has an embedding)
        embedding_path = UPLOAD_DIR / f"{request.voice_id}_embedding.pt"
        if embedding_path.exists():
            # Load the custom speaker embedding
            target_se = torch.load(embedding_path)
            
            # Generate base speech
            src_path = UPLOAD_DIR / "temp.wav"
            base_speaker_tts.tts(
                text=request.text,
                output_path=str(src_path),
                speaker='default',
                language='English',
                speed=1.0
            )
            
            # Convert to target voice
            output_path = UPLOAD_DIR / f"generated_{request.voice_id}.wav"
            source_se = torch.load(str(OPENVOICE_DIR / "checkpoints" / "base_speakers" / "EN" / "en_default_se.pth")).to(target_se.device)
            
            tone_color_converter.convert(
                audio_src_path=str(src_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_path),
                message="@MyShell"
            )
        else:
            # Use the base model with predefined voice
            output_path = UPLOAD_DIR / f"generated_{request.voice_id}.wav"
            base_speaker_tts.tts(
                text=request.text,
                output_path=str(output_path),
                speaker=request.voice_id,
                language='English',
                speed=1.0
            )
            
        logger.info(f"Successfully generated speech: {output_path}")
        
        # Read the generated audio file
        with open(output_path, "rb") as f:
            audio_data = f.read()
            
        return StreamingResponse(io.BytesIO(audio_data), media_type="audio/wav")
    except Exception as e:
        logger.error(f"Error in generate_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if src_path and src_path.exists():
            try:
                src_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up src_path: {e}")
        if output_path and output_path.exists():
            try:
                output_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up output_path: {e}")

@app.get("/voices")
async def list_voices():
    try:
        if base_speaker_tts is None:
            raise HTTPException(
                status_code=500, 
                detail="OpenVoice models not initialized. Please check the server logs for details."
            )
        
        # Get available speakers from the model's hps
        voices = [
            {"id": voice_id, "name": voice_id}
            for voice_id in base_speaker_tts.hps.speakers.keys()
        ]
        logger.info(f"Available voices: {voices}")
        
        return {
            "voices": voices
        }
    except Exception as e:
        logger.error(f"Error in list_voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 