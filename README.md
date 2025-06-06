# InsightAI Project

A full-stack application combining a React frontend with a Node.js/Express backend for Content Creation.

## Prerequisites

Before you begin, ensure you have the following installed:
- Node.js (v18.0.0 or higher)
- npm (v9.0.0 or higher)
- Python (v3.8 or higher)
- Git
- FFmpeg (for audio processing)
- CUDA-capable GPU (recommended for OpenVoice and Whisper services)

## Project Structure
```
insight-ai-project/
├── frontend/          # React frontend application
├── backend/           # Node.js/Express backend server
│   ├── whisper_service/  # Audio transcription service
│   └── openvoice_service/ # Voice cloning service
└── README.md         # Project documentation
```

## Environment Setup

### Backend Environment Variables
Create a `.env` file in the `backend` directory with the following variables:

```env
# Server Configuration
PORT=5000
NODE_ENV=development

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/insight-ai
# or your MongoDB Atlas URI
# MONGODB_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/insight-ai

# Authentication
JWT_SECRET=your_jwt_secret_key
JWT_EXPIRES_IN=7d

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Service URLs
WHISPER_SERVICE_URL=http://localhost:8000
OPENVOICE_SERVICE_URL=http://localhost:8001

# Optional: YouTube API (if using YouTube features)
YOUTUBE_API_KEY=your_youtube_api_key

# Optional: AWS Configuration (if using AWS services)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_bucket_name
```

### Frontend Environment Variables
Create a `.env` file in the `frontend` directory with the following variables:

```env
# API Configuration
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000

# Optional: Analytics (if using analytics)
REACT_APP_GA_TRACKING_ID=your_ga_tracking_id

# Optional: Feature Flags
REACT_APP_ENABLE_VOICE_CLONING=true
REACT_APP_ENABLE_TRANSCRIPTION=true

# Optional: Third-party Services
REACT_APP_STRIPE_PUBLIC_KEY=your_stripe_public_key
```

### Python Services Environment Variables

#### Whisper Service
Create a `.env` file in the `backend/whisper_service` directory:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Model Configuration
WHISPER_MODEL=base
DEVICE=cuda  # or cpu if no GPU available
```

#### OpenVoice Service
Create a `.env` file in the `backend/openvoice_service` directory:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8001

# Model Configuration
DEVICE=cuda  # or cpu if no GPU available
MODEL_PATH=./OpenVoice/checkpoints
```

## Installation

### Python Services Setup

#### 1. Install FFmpeg
- **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH
- **Linux**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`

#### 2. Set up Whisper Service
1. Navigate to the whisper service directory:
   ```bash
   cd backend/whisper_service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the service:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

#### 3. Set up OpenVoice Service
1. Navigate to the OpenVoice service directory:
   ```bash
   cd backend/openvoice_service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Install OpenVoice and its dependencies:
   ```bash
   # Clone OpenVoice repository
   git clone https://github.com/myshell-ai/OpenVoice.git
   cd OpenVoice
   pip install -e .
   cd ..
   
   # Install additional requirements
   pip install fastapi uvicorn python-multipart torch torchaudio numpy librosa soundfile
   ```

4. Download model checkpoints:
   ```bash
   python download_checkpoints.py
   ```

5. Start the service:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the backend server:
   ```bash
   node index.js
   ```

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## Development

### Running the Application
1. Start the Python services:
   ```bash
   # Terminal 1 - Whisper Service
   cd backend/whisper_service
   uvicorn main:app --host 0.0.0.0 --port 8000

   # Terminal 2 - OpenVoice Service
   cd backend/openvoice_service
   uvicorn main:app --host 0.0.0.0 --port 8001
   ```

2. Start the backend server (from the backend directory):
   ```bash
   node index.js
   ```

3. Start the frontend development server (from the frontend directory):
   ```bash
   npm start
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- Whisper Service: http://localhost:8000
- OpenVoice Service: http://localhost:8001

### Building for Production
1. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

2. The production build will be available in the `frontend/build` directory.

## API Documentation

### Endpoints

#### Authentication
- POST `/api/auth/register` - Register a new user
- POST `/api/auth/login` - Login user
- GET `/api/auth/profile` - Get user profile

#### Insights
- GET `/api/insights` - Get all insights
- POST `/api/insights` - Create new insight
- GET `/api/insights/:id` - Get insight by ID
- PUT `/api/insights/:id` - Update insight
- DELETE `/api/insights/:id` - Delete insight

#### Whisper Service
- POST `/transcribe` - Transcribe audio to text
- POST `/transcribe-youtube` - Transcribe YouTube video

#### OpenVoice Service
- POST `/clone-voice` - Clone voice from audio sample
- POST `/generate-speech` - Generate speech with cloned voice
- GET `/voices` - List available voices

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email tanzeel1256@gmail.com or create an issue in the repository.

## Acknowledgments

- React.js
- Node.js
- Express.js
- MongoDB
- OpenAI API
- OpenAI Whisper
- OpenVoice
