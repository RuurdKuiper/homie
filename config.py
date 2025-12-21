#!/usr/bin/env python3
"""Configuration constants and paths for Homie voice assistant."""

import os

# =============================
# Audio Configuration
# =============================
RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # 480 samples
RESPEAKER_INDEX = 0  # Index from your logs
VAD_MODE = 3          # 0 (most sensitive) to 3 (least sensitive)
SILENCE_TIMEOUT = 1.2 # seconds of silence to stop listening
MAX_LISTENING_TIME = 20.0 # seconds
ROLLBACK_BUFFER_SIZE = int(1.0 * RATE / FRAME_SIZE)  # 3 seconds of frames to keep as history
SILENCE_RMS_THRESHOLD = 0.4  # start here, tune between 0.01–0.03

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
WHISPER_MODEL_SIZE = "tiny"  # Options: "tiny", "base", "small", "medium", "large"
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
# LLM_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
# LLM_URL = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"

# =============================
# Performance Settings
# =============================
SAVE_DEBUG_RECORDINGS = True  # Set to True to save audio recordings for debugging (slower)
LLM_MAX_TOKENS = 48  # Reduced from 64 for faster responses
LLM_TEMPERATURE = 0.6  # Slightly lower for faster, more deterministic generation

