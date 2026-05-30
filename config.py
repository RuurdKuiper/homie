#!/usr/bin/env python3
"""Configuration constants and paths for Homie voice assistant."""

import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# =============================
# Cloud Service Configuration (True = use cloud, False = use local)
# =============================
USE_CLOUD_STT = True          # Speech-to-Text: False = Whisper local, True = OpenAI Whisper API
USE_CLOUD_LLM = True          # LLM: False = llama.cpp local, True = OpenAI API
USE_CLOUD_TTS = True          # Text-to-Speech: False = Piper local, True = OpenAI TTS

# =============================
# OpenAI API Configuration
# =============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_STT_MODEL = "gpt-4o-mini-transcribe"  # Latest: gpt-4o-mini-transcribe for cloud STT
OPENAI_LLM_MODEL = "gpt-5.4-mini"            # Latest: gpt-5.4-mini for fast, efficient cloud LLM
OPENAI_TTS_MODEL = "gpt-4o-mini-tts"        # Latest: gpt-4o-mini-tts for cloud TTS
OPENAI_TTS_VOICE = "echo"                   # Options: alloy, echo, fable, onyx, nova, shimmer

# =============================
# Audio Configuration
# =============================
STT_LANGUAGE = "nl"          # Language code for STT: "nl" = Dutch, "en" = English
RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # 480 samples
RESPEAKER_INDEX = 0  # Index from your logs
VAD_MODE = 3          # 0 (most sensitive) to 3 (least sensitive)
SILENCE_TIMEOUT = 2.0 # seconds of silence to stop listening
MAX_LISTENING_TIME = 20.0 # seconds
ROLLBACK_BUFFER_SIZE = int(1.0 * RATE / FRAME_SIZE)  # 3 seconds of frames to keep as history
SILENCE_RMS_THRESHOLD = 0.6  # start here, tune between 0.01–0.03

# =============================
# Wake Word Configuration
# =============================
WAKE_WORD = "hey homie"

# =============================
# Path Configuration
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =============================
# Vosk Model (for wake word)
# =============================
VOSK_NAME = "vosk-model-small-en-us-0.15"
VOSK_PATH = os.path.join(MODEL_DIR, VOSK_NAME)
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"

# =============================
# Whisper Model (for command transcription)
# =============================
WHISPER_MODEL_SIZE = "base"  # Options: "tiny", "base", "small", "medium", "large"
# Use "tiny" or "base" for faster transcription (lower accuracy)
# Use "small" for balanced speed/accuracy (default)

# =============================
# Piper TTS
# =============================
PIPER_DIR = os.path.join(MODEL_DIR, "piper")
PIPER_VOICE_EN = os.path.join(PIPER_DIR, "en_US-amy-medium.onnx")
PIPER_VOICE_NL = os.path.join(PIPER_DIR, "nl_NL-mls-medium.onnx")
PIPER_VOICE_DEFAULT = PIPER_VOICE_EN

# =============================
# LLM (llama.cpp)
# =============================
LLAMA_SERVER_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
# LLM_NAME = "phi-4-mini-instruct.Q4_K_M.gguf"
LLM_NAME = "Qwen3-1.7B-Q4_K_M.gguf"
LLM_PATH = os.path.join(MODEL_DIR, LLM_NAME)

# Large model for long answers
LLM_LONG_NAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
LLM_LONG_PATH = os.path.join(MODEL_DIR, LLM_LONG_NAME)
LLM_LONG_PORT = 8081  # Different port for the large model
LLM_LONG_PROCESS = None  # Will be set when server starts

# LLM_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
# LLM_URL = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"

# =============================
# Weather Configuration
# =============================
WEATHER_DEFAULT_LOCATION = "Utrecht,Netherlands"  # Default location for weather queries

# =============================
# Knowledge And Media Tools
# =============================
WIKIPEDIA_TIMEOUT = 5
WEB_SEARCH_TIMEOUT = 10
WEB_SEARCH_DEFAULT_RESULTS = 3
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
SPOTIFY_CACHE_PATH = os.path.join(BASE_DIR, ".spotify_cache")

# =============================
# Performance Settings
# =============================
SAVE_DEBUG_RECORDINGS = False  # Set to True to save audio recordings for debugging (slower)
LLM_MAX_TOKENS = 768  # Allows long spoken answers and detailed stories
LLM_LONG_MAX_TOKENS = 1024  # Extra budget for expansive long-form answers
LLM_TEMPERATURE = 0.6  # Slightly lower for faster, more deterministic generation
LLM_LONG_TEMPERATURE = 0.7  # Slightly higher for more creative long answers
CONVERSATION_MEMORY_TURNS = 8  # Number of user/assistant turns to keep for follow-up context

