import os
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def main():
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path("OpenVoice/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs for the model checkpoints
    checkpoint_urls = {
        "model.pt": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/model.pt",
        "config.json": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/config.json"
    }
    
    # Download each checkpoint file
    for filename, url in checkpoint_urls.items():
        destination = checkpoint_dir / filename
        if not destination.exists():
            logger.info(f"Downloading {filename}...")
            try:
                download_file(url, destination)
                logger.info(f"Successfully downloaded {filename}")
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
        else:
            logger.info(f"{filename} already exists, skipping...")

if __name__ == "__main__":
    main() 