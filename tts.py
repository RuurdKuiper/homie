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


def generate_audio(text, silence_ms=1000):
    """Generate audio from text using Piper TTS with prepended silence.
    
    Returns:
        tuple: (audio_float, sample_rate) where audio_float is ready for playback
    """
    if not text.strip():
        return None, None

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
        
        # 3) Convert to float32 for sounddevice
        audio_float = audio_with_silence.astype(np.float32) / 32768.0
        return audio_float, sr

    except Exception as e:
        print(f"[ERROR] Piper TTS generation failed: {e}", file=sys.stderr)
        return None, None


def play_audio(audio_float, sample_rate):
    """Play audio using sounddevice."""
    if audio_float is None:
        return
    try:
        sd.play(audio_float, samplerate=sample_rate, blocking=True)
    except Exception as e:
        print(f"[ERROR] Audio playback failed: {e}", file=sys.stderr)


def speak(text, silence_ms=1000):
    """Speak text using Piper TTS with prepended silence - optimized for in-memory processing.
    
    This is a convenience function that combines generate_audio and play_audio.
    For separate timing, use generate_audio() and play_audio() directly.
    """
    audio_float, sr = generate_audio(text, silence_ms)
    if audio_float is not None:
        play_audio(audio_float, sr)

