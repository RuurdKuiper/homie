#!/usr/bin/env python3
import os
import sys

# Disable ONNX GPU providers BEFORE importing faster_whisper
os.environ['ONNXRUNTIME_DISABLE_GPU'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import time
import queue
import tempfile
import zipfile
import urllib.request
import subprocess
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer
import soundfile as sf
import logging
import webrtcvad
import pyttsx3
import re

from piper import PiperVoice
from piper.config import SynthesisConfig
import wave

# Suppress sounddevice warnings
sd.default.latency = 'high'  # Use higher latency to reduce overflow risk

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech (default is 200)
tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)

# =============================
# Configuration
# =============================
RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)  # 480 samples
RESPEAKER_INDEX = 0  # Index from your logs
WAKE_WORD = "hey homie"
VAD_MODE = 3          # 0 (most sensitive) to 3 (least sensitive)
SILENCE_TIMEOUT = 1.0 # seconds of silence to stop listening
MAX_LISTENING_TIME = 20.0 # seconds
ROLLBACK_BUFFER_SIZE = int(1.5 * RATE / FRAME_SIZE)  # 3 seconds of frames to keep as history
SILENCE_RMS_THRESHOLD = 0.4  # start here, tune between 0.01–0.03

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ---- Vosk Model (for wake word) ----
VOSK_NAME = "vosk-model-small-en-us-0.15"
VOSK_PATH = os.path.join(MODEL_DIR, VOSK_NAME)
VOSK_URL = f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip"

# ---- Whisper Model (for command transcription) ----
WHISPER_MODEL_SIZE = "small"

# ---- Piper TTS ----
PIPER_DIR = os.path.join(MODEL_DIR, "piper")

PIPER_VOICE_EN = os.path.join(PIPER_DIR, "en_US-amy-medium.onnx")
PIPER_VOICE_NL = os.path.join(PIPER_DIR, "nl_NL-mls-medium.onnx")

PIPER_VOICE_DEFAULT = PIPER_VOICE_EN

print("[TTS] Loading Piper voice...")
piper_voice = PiperVoice.load(PIPER_VOICE_DEFAULT)

PIPER_SYN_CONFIG = SynthesisConfig(
    volume=1.0,
    length_scale=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=True,
)

# ---- LLM (llama.cpp) ----
LLAMA_SERVER_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
# LLM_NAME = "phi-4-mini-instruct.Q4_K_M.gguf"
LLM_NAME = "Qwen3-1.7B-Q4_K_M.gguf"
LLM_PATH = os.path.join(MODEL_DIR, LLM_NAME)
# LLM_URL = "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
# LLM_URL = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF/resolve/main/Phi-4-mini-instruct-Q4_K_M.gguf"

# Global audio queue - will be recreated when we need to flush it
audio_q = queue.Queue()

vad = webrtcvad.Vad(VAD_MODE)

def flush_audio_queue():
    """Flush all pending audio frames from the queue"""
    global audio_q
    # Create a new queue, discarding all old frames
    audio_q = queue.Queue()

# =============================
# Helpers (Auto-Download Logic)
# =============================
def is_speech(frame_int16):
    return vad.is_speech(frame_int16.tobytes(), RATE)

def ensure_vosk():
    if os.path.isdir(VOSK_PATH):
        return
    print(f"[INFO] Vosk model not found at {VOSK_PATH}. Downloading...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = VOSK_PATH + ".zip"
    urllib.request.urlretrieve(VOSK_URL, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(MODEL_DIR)
    os.remove(zip_path)
    print("[INFO] Vosk download complete.")

def audio_callback(indata, frames, time, status):
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

def ensure_llm_safe():
    if os.path.isfile(LLM_PATH):
        # Optional: check if file is too small (meaning it failed earlier)
        if os.path.getsize(LLM_PATH) > 1000000: # > 1MB
            return
            
    print(f"[INFO] Downloading large model to {LLM_PATH}...")
    os.makedirs(os.path.dirname(LLM_PATH), exist_ok=True)
    
    # -c: continue partial download
    # -O: output file path
    cmd = ["wget", "-c", LLM_URL, "-O", LLM_PATH]
    
    try:
        # We use run() here because we want the script to wait until it's done
        subprocess.run(cmd, check=True)
        print("[SUCCESS] Model downloaded.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Download failed. Try running it manually in terminal.")

def apply_agc(audio_int16, target_rms=0.1):
    audio = audio_int16.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    gain = (target_rms * 32768.0) / rms
    audio = audio * gain
    audio = np.clip(audio, -32768, 32767)
    return audio.astype(np.int16)

def apply_agc_frame(frame_int16, target_rms=0.08, max_gain=20.0):
    audio = frame_int16.astype(np.float32)
    rms = np.sqrt(np.mean(audio**2)) + 1e-8
    gain = min((target_rms * 32768.0) / rms, max_gain)
    audio *= 1.0
    return np.clip(audio, -32768, 32767).astype(np.int16)


def transcribe_audio(audio_data, whisper_model):
    """Transcribe audio data using faster-whisper"""
    try:
        # Create a temporary WAV file
        # with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        #     tmp_path = tmp.name
        #     sf.write(tmp_path, audio_data, RATE)
        
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
        
        # Clean up
        # os.unlink(tmp_path)
        return text
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
        return ""

def ensure_whisper_model():
    """Pre-download the Whisper model to avoid downloading during runtime"""
    print("[INFO] Checking Whisper model cache...")
    try:
        # Set cache to models folder
        cache_dir = os.path.join(MODEL_DIR, "whisper_cache")
        os.environ['HF_HOME'] = cache_dir
        
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[INFO] Whisper model is ready!")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load Whisper model: {e}", file=sys.stderr)
        print("[ERROR] Make sure you have an internet connection for the first run.")
        print("[ERROR] Or run: python setup_models.py")
        return None

# =============================
# LLM Logic
# =============================
llm_process = None # Global to handle cleanup

def start_llm_server():
    global llm_process
    print("[LLM] 🚀 Preloading model into RAM...")
    
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", LLM_PATH,
        "--port", "8080",
        "--n-gpu-layers", "0", # Pi uses CPU
        "--threads", "4",
        "--ctx-size", "512",  # Smaller context = faster on Pi
        "--log-disable"
    ]
    
    # Launch as background process
    llm_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to wake up
    print("[LLM] Waiting for server to be ready...", end="", flush=True)
    for _ in range(60): # Give it 30 seconds to load
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health") as f:
                if f.getcode() == 200:
                    print(" READY! ✅")
                    return True
        except:
            print(".", end="", flush=True)
            time.sleep(1)
    print(" TIMEOUT ❌")
    return False

def get_model_type():
    """Detect model type from LLM_NAME"""
    if "phi" in LLM_NAME.lower():
        return "phi"
    elif "qwen" in LLM_NAME.lower():
        return "qwen"
    else:
        return "phi"  # Default to Phi format

def format_prompt(prompt):
    """Format prompt based on model type"""
    model_type = get_model_type()
    system_msg = (
        "You are a home assistant. Respond with only one sentence."
    )
    
    if model_type == "qwen":
        # Qwen format: <|im_start|>role\ncontent<|im_end|>
        formatted = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n"
    else:  # Phi format
        # Phi-4-mini-instruct format: <|role|>content<|end|>
        formatted = f"<|system|>{system_msg}<|end|><|user|>{prompt}<|end|><|assistant|>"
    
    return formatted

def speak(text, silence_ms=2000):
    """Speak text using Piper TTS with prepended silence."""
    if not text.strip():
        return

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
            os.remove(wav_path)
        except Exception:
            pass


def clean_llm_output(text: str) -> str:
    # Remove <think>...</think> blocks (multiline safe)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove any remaining angle-bracket tokens like <think>, </think>, <s>, </s>
    text = re.sub(r"</?[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def run_llm(prompt):
    url = "http://127.0.0.1:8080/completion"
    formatted_prompt = format_prompt(prompt)
    
    data = json.dumps({
        "prompt": formatted_prompt,
        "n_predict": 64,
        "temperature": 0.7,
        "stream": True
    }).encode("utf-8")
    
    try:
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            response_text = ""
            # llama.cpp streams as Server-Sent Events (SSE) format
            for line in response:
                raw_line = line.decode('utf-8').strip()
                if not raw_line:
                    continue
                # Remove "data: " prefix if present
                if raw_line.startswith("data: "):
                    raw_line = raw_line[6:]  # Remove "data: "
                try:
                    res_json = json.loads(raw_line)
                    content = res_json.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        response_text += content
                except json.JSONDecodeError:
                    # Skip any non-JSON lines
                    pass
            return clean_llm_output(response_text)
            
    except Exception as e:
        print(f"(Error: {e})", flush=True)
        return ""

def frame_rms(frame_int16):
    return np.sqrt(np.mean(frame_int16.astype(np.float32) ** 2)) / 32768.0


# =============================
# Main
# =============================
def main():
    ensure_vosk()
    ensure_llm_safe()

    if not start_llm_server():
        print("Could not start LLM server. Check paths!")
        return

    print("[INFO] Loading Vosk model for wake word detection...")
    stt_model = Model(VOSK_PATH)
    wake_recognizer = KaldiRecognizer(stt_model, RATE, '["hey homie", "[unk]"]')
    
    print("[INFO] Loading Whisper model for command transcription...")
    whisper_model = ensure_whisper_model()
    if not whisper_model:
        print("[ERROR] Could not load Whisper model!")
        return
    
    state = "IDLE"
    last_partial = ""

    print("--- Available Audio Devices ---")
    print(sd.query_devices())
    print("-------------------------------")
    print(f"--- STARTING ON DEVICE {RESPEAKER_INDEX} ---")

    # Open Audio Stream
    with sd.InputStream(
        device=RESPEAKER_INDEX,
        samplerate=RATE,
        channels=2,        # ReSpeaker requires 2 channels
        dtype="int16",
        blocksize=FRAME_SIZE,
        callback=audio_callback,
    ):
        print(f'[READY] Say "{WAKE_WORD}"...')
        
        listening_audio = []
        wake_check_counter = 0  # Counter to reduce logging frequency
        silence_frames = 0  # Counter for consecutive silence frames
        rollback_buffer = []  # Keep last 2 seconds of audio for history
        
        while True:


            frame = audio_q.get()

            frame_i16 = frame.astype(np.int16)
            rms = frame_rms(frame_i16)
            print(f"\rRMS: {rms:.4f}   ", end="")

            # Always maintain a rolling buffer of recent audio
            rollback_buffer.append(frame.copy())
            if len(rollback_buffer) > ROLLBACK_BUFFER_SIZE:
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
                    if WAKE_WORD in final_text:
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
                # listening_audio.extend(frame)
                listening_audio.append(frame.copy())

                frame_i16 = frame.astype(np.int16)
                rms = frame_rms(frame_i16)

                if rms < SILENCE_RMS_THRESHOLD:
                    speech = False
                else:
                    speech = is_speech(frame_i16)


                if speech:
                    silence_frames = 0
                else:
                    silence_frames += 1

                silence_duration = silence_frames * FRAME_MS / 1000.0

                # Hard safety cap (still useful)
                max_duration = len(listening_audio) / RATE
                if silence_duration >= SILENCE_TIMEOUT or max_duration >= MAX_LISTENING_TIME:
                    print(f"[INFO] Silence duration: {silence_duration:.2f}s")
                    if max_duration >= MAX_LISTENING_TIME:
                        print("[INFO] Max listening time reached (10s).")

                    # listening_array = np.array(listening_audio, dtype=np.int16)
                    listening_array = np.concatenate(listening_audio).astype(np.int16)


                    # Save recording to file for debugging
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    recording_path = os.path.join(BASE_DIR, f"recording_{timestamp}.wav")
                    sf.write(recording_path, listening_array, RATE)
                    print(f"[DEBUG] Recording saved to {recording_path}")

                    print("[TRANSCRIBING]...")
                    user_text = transcribe_audio(listening_array, whisper_model)

                    if user_text.strip():
                        print(f"[USER] {user_text}")
                        print(f"[LLM] Processing prompt...")
                        print("[HOMIE] ", end="", flush=True)
                        response = run_llm(user_text)
                        print()
                        
                        # Speak the response
                        if response.strip():
                            print("[SPEAKING]...")
                            speak(response)

                    flush_audio_queue()
                    wake_recognizer.Reset()
                    listening_audio = []
                    silence_frames = 0
                    state = "IDLE"
                    last_partial = ""
                    print(f'[READY] Say "{WAKE_WORD}"...')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Goodbye.")