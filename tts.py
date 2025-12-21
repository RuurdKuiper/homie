#!/usr/bin/env python3
"""Text-to-speech using Piper TTS."""

import os
import sys
import tempfile
import subprocess
import wave
import numpy as np
import soundfile as sf
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
    """Speak text using Piper TTS with prepended silence."""
    if not text.strip():
        return

    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        # 1) Synthesize speech to WAV
        with wave.open(wav_path, "wb") as wav_file:
            piper_voice.synthesize_wav(
                text,
                wav_file,
                syn_config=PIPER_SYN_CONFIG
            )

        # 2) Read audio and prepend silence
        audio, sr = sf.read(wav_path, dtype="int16")
        silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.int16)
        audio = np.concatenate([silence, audio])

        # 3) Write back
        sf.write(wav_path, audio, sr)

        # 4) Play
        subprocess.run(["aplay", wav_path], check=False)

    except Exception as e:
        print(f"[ERROR] Piper TTS failed: {e}", file=sys.stderr)

    finally:
        try:
            if wav_path:
                os.remove(wav_path)
        except Exception:
            pass

