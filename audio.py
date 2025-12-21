#!/usr/bin/env python3
"""Audio stream handling, VAD, and AGC utilities."""

import sys
import queue
import numpy as np
import sounddevice as sd
import webrtcvad
import config

# Suppress sounddevice warnings
sd.default.latency = 'high'  # Use higher latency to reduce overflow risk

# Global audio queue - will be recreated when we need to flush it
audio_q = queue.Queue()

# Initialize VAD
vad = webrtcvad.Vad(config.VAD_MODE)


def flush_audio_queue():
    """Flush all pending audio frames from the queue"""
    global audio_q
    # Create a new queue, discarding all old frames
    audio_q = queue.Queue()


def is_speech(frame_int16):
    """Check if audio frame contains speech using VAD"""
    return vad.is_speech(frame_int16.tobytes(), config.RATE)


def audio_callback(indata, frames, time, status):
    """Callback function for audio stream input"""
    if status:
        # Log overflow/underflow but don't crash
        if "overflow" in str(status).lower():
            pass  # Silently ignore overflow - some frames will be lost but that's okay during processing
        else:
            print(f"[STATUS] {status}", file=sys.stderr)
    # Extract channel 0 (Mono) from the Stereo stream
    mono_data = indata[:, 0].copy()
    # Use non-blocking put to avoid hanging the audio thread
    try:
        audio_q.put_nowait(mono_data)
    except queue.Full:
        pass  # Queue is full, skip this frame to avoid blocking


def apply_agc(audio_int16, target_rms=0.1):
    """Apply automatic gain control to audio"""
    audio = audio_int16.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    gain = (target_rms * 32768.0) / rms
    audio = audio * gain
    audio = np.clip(audio, -32768, 32767)
    return audio.astype(np.int16)


def apply_agc_frame(frame_int16, target_rms=0.08, max_gain=20.0):
    """Apply automatic gain control to a single frame"""
    audio = frame_int16.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    gain = min((target_rms * 32768.0) / rms, max_gain)
    audio *= 1.0
    return np.clip(audio, -32768, 32767).astype(np.int16)


def frame_rms(frame_int16):
    """Calculate RMS (Root Mean Square) of an audio frame"""
    return np.sqrt(np.mean(frame_int16.astype(np.float32) ** 2)) / 32768.0

