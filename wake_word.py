#!/usr/bin/env python3
"""Wake word detection using Vosk."""

import os
import json
import zipfile
import urllib.request
from vosk import Model, KaldiRecognizer
import config


def ensure_vosk():
    """Ensure Vosk model is downloaded and available"""
    if os.path.isdir(config.VOSK_PATH):
        return
    print(f"[INFO] Vosk model not found at {config.VOSK_PATH}. Downloading...")
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    zip_path = config.VOSK_PATH + ".zip"
    urllib.request.urlretrieve(config.VOSK_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(config.MODEL_DIR)
    os.remove(zip_path)
    print("[INFO] Vosk download complete.")


def create_wake_recognizer():
    """Create and return a Vosk wake word recognizer"""
    stt_model = Model(config.VOSK_PATH)
    wake_recognizer = KaldiRecognizer(stt_model, config.RATE, f'["{config.WAKE_WORD}", "[unk]"]')
    return wake_recognizer

