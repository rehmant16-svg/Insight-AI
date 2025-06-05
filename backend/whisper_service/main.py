from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
from transcribe import process_youtube_video

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranscriptionRequest(BaseModel):
    url: str

@app.post("/transcribe")
async def transcribe_video(request: TranscriptionRequest):
    try:
        logger.info(f"Received transcription request for URL: {request.url}")
        
        # Process the video and get transcription
        transcription = process_youtube_video(request.url)
        
        if transcription is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to transcribe video"
            )
            
        return {
            "status": "success",
            "transcription": transcription
        }
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 