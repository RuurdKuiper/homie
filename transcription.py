#!/usr/bin/env python3
"""Speech transcription using Whisper."""

import os
import sys
import numpy as np
from faster_whisper import WhisperModel
import config


def ensure_whisper_model():
    """Pre-download the Whisper model to avoid downloading during runtime"""
    print("[INFO] Checking Whisper model cache...")
    try:
        # Set cache to models folder
        cache_dir = os.path.join(config.MODEL_DIR, "whisper_cache")
        os.environ['HF_HOME'] = cache_dir
        
        model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[INFO] Whisper model is ready!")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load Whisper model: {e}", file=sys.stderr)
        print("[ERROR] Make sure you have an internet connection for the first run.")
        print("[ERROR] Or run: python setup_models.py")
        return None


def transcribe_audio(audio_data, whisper_model):
    """Transcribe audio data using faster-whisper"""
    try:
        # Transcribe using faster-whisper with VAD (Voice Activity Detection)
        # VAD will filter out silence automatically, making transcription more accurate
        segments, info = whisper_model.transcribe(
            audio_data.astype(np.float32) / 32768.0,
            language="en", 
            beam_size=1,
            without_timestamps=True,
            best_of=1,
            vad_filter=True  # Enable built-in VAD for better speech detection
        )
        text = "".join([seg.text for seg in segments]).strip()
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
        return ""

