#!/usr/bin/env python3
"""Text-to-speech using Piper TTS."""

import os
import sys
import io
import wave
import numpy as np
import soundfile as sf
import sounddevice as sd
from piper import PiperVoice
from piper.config import SynthesisConfig
import config

# Initialize Piper voice
print("[TTS] Loading Piper voice...")
piper_voice = PiperVoice.load(config.PIPER_VOICE_DEFAULT)

PIPER_SYN_CONFIG = SynthesisConfig(
    volume=1.0,
    length_scale=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=True,
)


def speak(text, silence_ms=2000):
    """Speak text using Piper TTS with prepended silence - optimized for in-memory processing."""
    if not text.strip():
        return

    try:
        # Use in-memory buffer instead of file I/O for faster processing
        # 1) Synthesize speech to in-memory WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            piper_voice.synthesize_wav(
                text,
                wav_file,
                syn_config=PIPER_SYN_CONFIG
            )
        
        # 2) Read audio from buffer and prepend silence
        wav_buffer.seek(0)
        audio, sr = sf.read(wav_buffer, dtype="int16")
        silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.int16)
        audio_with_silence = np.concatenate([silence, audio])
        
        # 3) Play directly from memory using sounddevice (no file I/O)
        # Convert to float32 for sounddevice
        audio_float = audio_with_silence.astype(np.float32) / 32768.0
        sd.play(audio_float, samplerate=sr, blocking=True)

    except Exception as e:
        print(f"[ERROR] Piper TTS failed: {e}", file=sys.stderr)

