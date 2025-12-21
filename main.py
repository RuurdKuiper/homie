#!/usr/bin/env python3
"""Main entry point for Homie voice assistant."""

import os
import sys
import json
import time
import sounddevice as sd
import soundfile as sf
import numpy as np

# Disable ONNX GPU providers BEFORE importing faster_whisper
os.environ['ONNXRUNTIME_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import config
import audio
import wake_word
import transcription
import llm
import tts
import models


def main():
    """Main application loop with state machine"""
    # Ensure models are available
    wake_word.ensure_vosk()
    models.ensure_llm_safe()

    # Start LLM server
    if not llm.start_llm_server():
        print("Could not start LLM server. Check paths!")
        return

    # Load models
    print("[INFO] Loading Vosk model for wake word detection...")
    wake_recognizer = wake_word.create_wake_recognizer()
    
    print("[INFO] Loading Whisper model for command transcription...")
    whisper_model = transcription.ensure_whisper_model()
    if not whisper_model:
        print("[ERROR] Could not load Whisper model!")
        return
    
    # State machine
    state = "IDLE"
    last_partial = ""

    print("--- Available Audio Devices ---")
    print(sd.query_devices())
    print("-------------------------------")
    print(f"--- STARTING ON DEVICE {config.RESPEAKER_INDEX} ---")

    # Open Audio Stream
    with sd.InputStream(
        device=config.RESPEAKER_INDEX,
        samplerate=config.RATE,
        channels=2,        # ReSpeaker requires 2 channels
        dtype="int16",
        blocksize=config.FRAME_SIZE,
        callback=audio.audio_callback,
    ):
        print(f'[READY] Say "{config.WAKE_WORD}"...')
        
        listening_audio = []
        wake_check_counter = 0  # Counter to reduce logging frequency
        silence_frames = 0  # Counter for consecutive silence frames
        rollback_buffer = []  # Keep last 2 seconds of audio for history
        
        while True:
            frame = audio.audio_q.get()

            frame_i16 = frame.astype(np.int16)
            rms = audio.frame_rms(frame_i16)
            print(f"\rRMS: {rms:.4f}   ", end="")

            # Always maintain a rolling buffer of recent audio
            rollback_buffer.append(frame.copy())
            if len(rollback_buffer) > config.ROLLBACK_BUFFER_SIZE:
                rollback_buffer.pop(0)

            
            if state == "IDLE":
                # Use Vosk for wake word detection
                # Feed audio continuously to Vosk (maintains internal state)
                raw_bytes = frame.astype(np.int16).tobytes()

                if wake_recognizer.AcceptWaveform(raw_bytes):
                    # Got a final result
                    res = json.loads(wake_recognizer.Result())
                    final_text = res.get("text", "").lower()
                    # Only trigger on final result for "hey homie"
                    if config.WAKE_WORD in final_text:
                        print(f"\n[WAKE] 👂 Match found!")
                        state = "LISTENING"

                        # 🔁 preload rollback audio
                        listening_audio = [f.copy() for f in rollback_buffer]
                        silence_frames = 0
                        wake_recognizer.Reset()
                        
                        wake_check_counter = 0
                        print("[LISTENING] Go ahead...")
                else:
                    # Partial result - only show it every ~1 second to avoid spam
                    wake_check_counter += 1
                    if wake_check_counter >= 33:  # ~33 frames * 30ms ≈ 1 second
                        wake_check_counter = 0
                        partial = json.loads(wake_recognizer.PartialResult())
                        partial_text = partial.get("partial", "").lower()
                        if partial_text and partial_text != last_partial:
                            print(f"\r[HEARING]: {partial_text: <30}", end="", flush=True)
                            last_partial = partial_text

            elif state == "LISTENING":
                listening_audio.append(frame.copy())

                frame_i16 = frame.astype(np.int16)
                rms = audio.frame_rms(frame_i16)

                if rms < config.SILENCE_RMS_THRESHOLD:
                    speech = False
                else:
                    speech = audio.is_speech(frame_i16)

                if speech:
                    silence_frames = 0
                else:
                    silence_frames += 1

                silence_duration = silence_frames * config.FRAME_MS / 1000.0

                # Hard safety cap (still useful)
                max_duration = len(listening_audio) / config.RATE
                if silence_duration >= config.SILENCE_TIMEOUT or max_duration >= config.MAX_LISTENING_TIME:
                    print(f"[INFO] Silence duration: {silence_duration:.2f}s")
                    if max_duration >= config.MAX_LISTENING_TIME:
                        print("[INFO] Max listening time reached (20s).")

                    listening_array = np.concatenate(listening_audio).astype(np.int16)

                    # Save recording to file for debugging (optional, disabled by default for performance)
                    if config.SAVE_DEBUG_RECORDINGS:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        recording_path = os.path.join(config.BASE_DIR, f"recording_{timestamp}.wav")
                        sf.write(recording_path, listening_array, config.RATE)
                        print(f"[DEBUG] Recording saved to {recording_path}")

                    print("[TRANSCRIBING]...")
                    user_text = transcription.transcribe_audio(listening_array, whisper_model)

                    if user_text.strip():
                        print(f"[USER] {user_text}")
                        print(f"[LLM] Processing prompt...")
                        print("[HOMIE] ", end="", flush=True)
                        response = llm.run_llm(user_text)
                        print()
                        
                        # Speak the response
                        if response.strip():
                            print("[SPEAKING]...")
                            tts.speak(response)

                    audio.flush_audio_queue()
                    wake_recognizer.Reset()
                    listening_audio = []
                    silence_frames = 0
                    state = "IDLE"
                    last_partial = ""
                    print(f'[READY] Say "{config.WAKE_WORD}"...')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Goodbye.")

