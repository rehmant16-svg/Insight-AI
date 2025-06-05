const express = require('express');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const cors = require('cors');
const fetch = require('node-fetch');
const { YoutubeTranscript } = require('youtube-transcript');
const User = require('./models/User');
require('dotenv').config();

const app = express();

// Enable CORS
app.use(cors());

// Parse incoming JSON bodies
app.use(express.json());

// If you need URL-encoded form data:
app.use(express.urlencoded({ extended: true }));

// Comment out MongoDB connection for now since it's not needed for transcription
/*
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log('Connected to MongoDB'))
  .catch((err) => console.error('MongoDB connection error:', err));
*/

app.get('/', (req, res) => {
  res.send('Hello World');
});


app.post('/create-user', async (req, res) => {
  try {
    const newUser = new User({ email: 'test@example.com' });
    await newUser.save();
    res.send('User created successfully');
  } catch (err) {
    res.status(500).send('Error creating user');
  }
});

app.post('/register', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).send('User already exists');
    }

    // Hash the password
    const hashedPassword = await bcrypt.hash(password, 10);

    // Create and save the new user
    const newUser = new User({ email, password: hashedPassword });
    await newUser.save();

    res.status(201).send('User registered successfully');
  } catch (err) {
    console.error(err);
    res.status(500).send('Error registering user');
  }
});

app.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find the user by email
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).send('Invalid email or password');
    }

    // Compare the provided password with the stored hash
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).send('Invalid email or password');
    }

    // Generate a JWT
    const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET, { expiresIn: '1h' });

    res.json({ token });
  } catch (err) {
    res.status(500).send('Error logging in');
  }
});

const authMiddleware = (req, res, next) => {
  const token = req.header('Authorization')?.replace('Bearer ', '');

  if (!token) {
    return res.status(401).send('Access denied. No token provided.');
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    res.status(400).send('Invalid token');
  }
};

app.get('/protected', authMiddleware, (req, res) => {
  res.send('This is a protected route');
});

// Function to clean up transcript text
function cleanTranscriptText(text) {
  return text
    // Replace HTML entities
    .replace(/&amp;#39;/g, "'")
    .replace(/&amp;quot;/g, '"')
    .replace(/&amp;/g, '&')
    .replace(/&#39;/g, "'")
    .replace(/&quot;/g, '"')
    // Add proper spacing after punctuation
    .replace(/([.!?])(\w)/g, '$1 $2')
    // Remove extra spaces
    .replace(/\s+/g, ' ')
    .trim();
}

// Add the transcription endpoint
app.post('/api/transcribe', async (req, res) => {
  try {
    const { url } = req.body;
    
    // Extract video ID from URL
    const videoId = url.match(/(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^"&?\/\s]{11})/i)?.[1];
    
    if (!videoId) {
      return res.status(400).json({ error: 'Invalid YouTube URL' });
    }

    // Get transcript with language preference
    let transcripts;
    try {
      // First try to get English transcript
      transcripts = await YoutubeTranscript.fetchTranscript(videoId, {
        lang: 'en'
      });
    } catch (err) {
      // If English not available, get auto-generated transcript
      transcripts = await YoutubeTranscript.fetchTranscript(videoId);
    }
    
    // Process transcripts to create well-formatted text
    const processedTranscripts = transcripts.map(part => ({
      ...part,
      text: cleanTranscriptText(part.text)
    }));

    // Combine transcript parts with proper spacing and punctuation
    let fullTranscript = '';
    processedTranscripts.forEach((part, index) => {
      const text = part.text;
      // Add the text with proper spacing
      if (index > 0 && !text.match(/^[.,!?;]/) && !fullTranscript.match(/[.,!?;]$/)) {
        fullTranscript += ' ';
      }
      fullTranscript += text;
    });

    // Final cleanup of the full transcript
    fullTranscript = cleanTranscriptText(fullTranscript);

    res.json({ 
      transcript: fullTranscript,
      segments: processedTranscripts // Including individual segments with timestamps
    });
  } catch (error) {
    console.error('Transcription error:', error);
    res.status(500).json({ 
      error: 'Failed to get transcript',
      details: error.message 
    });
  }
});

// Add chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, transcript } = req.body;

    console.log('Received chat request:', { message, hasTranscript: !!transcript });

    // Check if Ollama is running
    try {
      const ollamaCheck = await fetch('http://localhost:11434/api/tags');
      if (!ollamaCheck.ok) {
        throw new Error('Ollama server is not responding');
      }
    } catch (error) {
      console.error('Ollama connection error:', error);
      return res.status(500).json({
        error: 'Failed to connect to Ollama',
        details: 'Please make sure Ollama is running on port 11434'
      });
    }

    // Make request to local Ollama instance
    console.log('Sending request to Ollama...');
    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: "deepseek-r1:32b",
        prompt: transcript === 'No transcript available' 
          ? `You are a helpful AI assistant. Please respond to: ${message}`
          : `You are an expert content creator and script writer assistant. You have access to a video transcript and can help create engaging content, scripts, and provide creative insights.

                Context: You have a video transcript that you can reference and analyze.
                
                Transcript: ${transcript}
                
                User Query: ${message}
                
                Please provide detailed, creative assistance focusing on:
                1. Content structure and flow
                2. Engaging storytelling elements
                3. Key points and highlights
                4. Audience engagement tips
                5. Script improvements or variations
                
                If the query is about specific content, reference relevant parts of the transcript.
                If it's a creative request, provide structured, actionable suggestions.`,
        stream: false,
        options: {
          temperature: 0.7,
          top_k: 50,
          top_p: 0.7,
          context_length: 2048
        }
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Ollama API error:', errorText);
      throw new Error(`Ollama API error: ${errorText}`);
    }

    const data = await response.json();
    console.log('Received response from Ollama');

    // Clean the response by removing <think> tags and their content
    const cleanResponse = data.response.replace(/<think>[\s\S]*?<\/think>/g, '').trim();

    res.json({ response: cleanResponse });
  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ 
      error: 'Failed to get AI response',
      details: error.message,
      hint: 'Make sure Ollama is running and the tinyllama model is installed using: ollama pull tinyllama'
    });
  }
});

// Add new endpoint for YouTube transcription
app.post('/api/transcribe-youtube', async (req, res) => {
  try {
    const { url } = req.body;
    
    if (!url) {
      return res.status(400).json({ error: 'URL is required' });
    }

    // Call the Whisper service
    const response = await fetch('http://localhost:8001/transcribe', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to transcribe video');
    }

    const data = await response.json();
    res.json(data);
  } catch (error) {
    console.error('Transcription error:', error);
    res.status(500).json({ 
      error: 'Failed to transcribe video',
      details: error.message
    });
  }
});

app.listen(process.env.PORT || 5000, () => {
  console.log(`Server running on port ${process.env.PORT || 5000}`);
});