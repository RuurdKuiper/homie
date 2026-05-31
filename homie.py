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
import re
from io import BytesIO
import shutil
import signal
import threading
import hashlib
from audio_output import choose_output_sink, ensure_output_sink, list_audio_sinks

from piper import PiperVoice
from piper.config import SynthesisConfig
import wave

# Import configuration
from config import (
    USE_CLOUD_STT, USE_CLOUD_LLM, USE_CLOUD_TTS,
    STT_LANGUAGE,
    SAVE_DEBUG_RECORDINGS,
    CONVERSATION_MEMORY_TURNS,
    LLM_MAX_TOKENS,
    RATE, FRAME_MS, FRAME_SIZE,
    RESPEAKER_INDEX, WAKE_WORD,
    VAD_MODE, SILENCE_TIMEOUT, MAX_LISTENING_TIME,
    ROLLBACK_BUFFER_SIZE, SILENCE_RMS_THRESHOLD,
    PIPER_SPEECH_SPEED, OPENAI_TTS_SPEED,
    TTS_CHUNK_MAX_CHARS, TTS_PREFETCH_CHUNKS,
    TTS_CACHE_DIR, TOOL_SPOKEN_PROMPTS,
    OPENAI_API_KEY, OPENAI_STT_MODEL, OPENAI_LLM_MODEL, 
    OPENAI_TTS_MODEL, OPENAI_TTS_VOICE
)
from tools.search import search_web, search_wikipedia
from tools.spotify import spotify_play

# Import OpenAI client if using cloud services
if USE_CLOUD_STT or USE_CLOUD_LLM or USE_CLOUD_TTS:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# =============================
# Tool Definitions for LLM
# =============================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for (e.g., 'Amsterdam, Netherlands')"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a specific time",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder message"
                    },
                    "minutes_from_now": {
                        "type": "integer",
                        "description": "How many minutes from now to remind (default: 5)"
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_device",
            "description": "Control smart home devices (lights, fans, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {
                        "type": "string",
                        "description": "The device name (e.g., 'bedroom light', 'living room fan')"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["on", "off", "toggle"],
                        "description": "The action to perform"
                    }
                },
                "required": ["device", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Search Wikipedia for reliable background knowledge on science, history, art, people, places, and concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic to look up on Wikipedia"
                    },
                    "max_sentences": {
                        "type": "integer",
                        "description": "How many summary sentences to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current or recent information such as news, football results, sports standings, release dates, and live updates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The web search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "How many results to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spotify_play",
            "description": "Play a song, artist, album, or playlist on Spotify.",
            "parameters": {
                "type": "object",
                "properties": {
                    "track_or_artist": {
                        "type": "string",
                        "description": "The song, artist, album, or playlist to play"
                    },
                    "context": {
                        "type": "string",
                        "enum": ["track", "artist", "album", "playlist"],
                        "description": "What kind of Spotify item to search for"
                    }
                },
                "required": ["track_or_artist"]
            }
        }
    }
]

# Suppress sounddevice warnings
sd.default.latency = 'high'  # Use higher latency to reduce overflow risk

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

piper_voice = None  # Will be initialized after downloading
conversation_history = []
speech_worker_thread = None
speech_request_queue = queue.Queue()
speech_stop_event = threading.Event()
speech_active_event = threading.Event()
playback_process = None
playback_lock = threading.Lock()

PIPER_SYN_CONFIG = SynthesisConfig(
    volume=1.0,
    length_scale=1.0 / max(PIPER_SPEECH_SPEED, 0.1),
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


def _spawn_with_pulseaudio(command):
    """Start playback command with the chosen Pulse sink if available."""
    sink = ensure_output_sink()
    env = os.environ.copy()
    if sink:
        env["PULSE_SINK"] = sink
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )


def start_audio_playback(path):
    """Start local audio playback while honoring the selected output sink."""
    extension = os.path.splitext(path)[1].lower()

    if extension == ".wav" and shutil.which("paplay"):
        sink = ensure_output_sink()
        command = ["paplay"]
        if sink:
            command.extend(["--device", sink])
        command.append(path)
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return _spawn_with_pulseaudio(["ffplay", "-nodisp", "-autoexit", path])


def wait_for_playback(process, stop_event):
    """Wait for playback to finish, terminating it if interrupted."""
    global playback_process

    with playback_lock:
        playback_process = process

    try:
        while process.poll() is None:
            if stop_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False
            time.sleep(0.05)
        return process.returncode == 0
    finally:
        with playback_lock:
            if playback_process is process:
                playback_process = None


def play_audio_file(path):
    """Play a local audio file synchronously while honoring the selected output sink."""
    process = start_audio_playback(path)
    if process is None:
        return
    process.wait()


def play_feedback_beep(frequency_hz, duration_ms, volume=0.2):
    """Play a short synthesized beep for UI feedback."""
    sample_count = max(1, int(RATE * duration_ms / 1000))
    time_axis = np.arange(sample_count, dtype=np.float32) / RATE
    envelope = np.linspace(1.0, 0.0, sample_count, dtype=np.float32)
    waveform = np.sin(2 * np.pi * frequency_hz * time_axis) * envelope * volume
    audio = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        beep_path = tmp.name

    try:
        sf.write(beep_path, audio, RATE)
        play_audio_file(beep_path)
    finally:
        try:
            os.remove(beep_path)
        except OSError:
            pass


def split_text_for_tts(text, max_chars=260):
    """Split long text into smaller TTS chunks to reduce first-audio latency."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), max_chars):
                chunks.append(sentence[start:start + max_chars].strip())
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def is_speaking_active():
    return speech_active_event.is_set() or not speech_request_queue.empty()


def stop_speaking():
    """Stop any active TTS generation or playback."""
    global playback_process

    speech_stop_event.set()

    while True:
        try:
            speech_request_queue.get_nowait()
        except queue.Empty:
            break

    with playback_lock:
        process = playback_process
    if process and process.poll() is None:
        process.terminate()

    with playback_lock:
        process = playback_process
        if process and process.poll() is None:
            process.kill()
        playback_process = None


def _build_local_tts_file(text, silence_ms):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    with wave.open(wav_path, "wb") as wav_file:
        piper_voice.synthesize_wav(
            text,
            wav_file,
            syn_config=PIPER_SYN_CONFIG
        )

    if silence_ms > 0:
        audio, sr = sf.read(wav_path, dtype="int16")
        silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.int16)
        audio = np.concatenate([silence, audio])
        sf.write(wav_path, audio, sr)

    return wav_path


def _build_cloud_tts_file(text):
    request_kwargs = {
        "model": OPENAI_TTS_MODEL,
        "voice": OPENAI_TTS_VOICE,
        "input": text,
    }
    if OPENAI_TTS_SPEED and OPENAI_TTS_SPEED != 1.0:
        request_kwargs["speed"] = OPENAI_TTS_SPEED

    try:
        response = openai_client.audio.speech.create(**request_kwargs)
    except Exception as exc:
        if "speed" in request_kwargs and "speed" in str(exc).lower():
            request_kwargs.pop("speed", None)
            response = openai_client.audio.speech.create(**request_kwargs)
        else:
            raise

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(response.content)

    return tmp_path


def get_tool_prompt_text(tool_name):
    return TOOL_SPOKEN_PROMPTS.get(tool_name, "Ik kijk dat meteen voor je na.")


def get_tts_cache_path(cache_key, extension):
    os.makedirs(TTS_CACHE_DIR, exist_ok=True)
    backend = f"cloud_{OPENAI_TTS_MODEL}_{OPENAI_TTS_VOICE}_{OPENAI_TTS_SPEED}" if USE_CLOUD_TTS else f"local_piper_{PIPER_SPEECH_SPEED}"
    hashed = hashlib.sha1(f"{backend}:{cache_key}".encode("utf-8")).hexdigest()[:16]
    return os.path.join(TTS_CACHE_DIR, f"{cache_key}_{hashed}.{extension}")


def ensure_cached_prompt_audio(tool_name):
    prompt_text = get_tool_prompt_text(tool_name)
    extension = "mp3" if USE_CLOUD_TTS else "wav"
    cache_path = get_tts_cache_path(tool_name, extension)
    if os.path.exists(cache_path):
        return cache_path

    temp_path = _build_cloud_tts_file(prompt_text) if USE_CLOUD_TTS else _build_local_tts_file(prompt_text, 0)
    try:
        shutil.move(temp_path, cache_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return cache_path


def warm_tool_prompt_cache():
    for tool_name in TOOL_SPOKEN_PROMPTS:
        if speech_stop_event.is_set():
            return
        try:
            ensure_cached_prompt_audio(tool_name)
        except Exception as exc:
            print(f"[WARN] Could not cache prompt for {tool_name}: {exc}")


def start_tool_prompt_cache_warmup():
    threading.Thread(target=warm_tool_prompt_cache, daemon=True).start()


def _enqueue_generated_audio(audio_queue, audio_path, stop_event):
    while not stop_event.is_set():
        try:
            audio_queue.put(audio_path, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _process_text_speech(text, silence_ms, stop_event):
    chunks = split_text_for_tts(text, max_chars=TTS_CHUNK_MAX_CHARS)
    if not chunks:
        return

    audio_queue = queue.Queue(maxsize=max(1, TTS_PREFETCH_CHUNKS))
    producer_done = threading.Event()
    producer_error = []
    sentinel = object()

    def producer():
        try:
            for index, chunk in enumerate(chunks):
                if stop_event.is_set():
                    return

                if USE_CLOUD_TTS:
                    print(f"[TTS] Generating chunk {index + 1}/{len(chunks)} with OpenAI TTS...")
                    audio_path = _build_cloud_tts_file(chunk)
                else:
                    audio_path = _build_local_tts_file(chunk, silence_ms if index == 0 else 0)

                if not _enqueue_generated_audio(audio_queue, audio_path, stop_event):
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    return
        except Exception as exc:
            producer_error.append(exc)
        finally:
            producer_done.set()
            while True:
                try:
                    audio_queue.put(sentinel, timeout=0.1)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        break

    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()

    try:
        while True:
            if stop_event.is_set() and producer_done.is_set() and audio_queue.empty():
                break
            try:
                item = audio_queue.get(timeout=0.1)
            except queue.Empty:
                if producer_done.is_set():
                    break
                continue

            if item is sentinel:
                break

            try:
                process = start_audio_playback(item)
                if process is None:
                    return
                if not wait_for_playback(process, stop_event):
                    return
            finally:
                if os.path.exists(item):
                    try:
                        os.remove(item)
                    except OSError:
                        pass
    finally:
        producer_thread.join(timeout=1)
        while True:
            try:
                leftover = audio_queue.get_nowait()
            except queue.Empty:
                break
            if leftover is sentinel:
                continue
            if os.path.exists(leftover):
                try:
                    os.remove(leftover)
                except OSError:
                    pass

    if producer_error:
        raise producer_error[0]


def _process_audio_prompt(path, stop_event):
    process = start_audio_playback(path)
    if process is None:
        return
    wait_for_playback(process, stop_event)


def _speech_queue_worker():
    while True:
        job = speech_request_queue.get()
        if job is None:
            return

        speech_stop_event.clear()
        speech_active_event.set()
        try:
            if job["kind"] == "audio":
                _process_audio_prompt(job["path"], speech_stop_event)
            else:
                _process_text_speech(job["text"], job.get("silence_ms", 0), speech_stop_event)
        except Exception as exc:
            print(f"[ERROR] Speech job failed: {exc}", file=sys.stderr)
        finally:
            speech_active_event.clear()


def ensure_speech_worker():
    global speech_worker_thread
    if speech_worker_thread is None or not speech_worker_thread.is_alive():
        speech_worker_thread = threading.Thread(target=_speech_queue_worker, daemon=True)
        speech_worker_thread.start()


def enqueue_tool_prompt(tool_name):
    ensure_speech_worker()
    try:
        cache_path = ensure_cached_prompt_audio(tool_name)
        speech_request_queue.put({"kind": "audio", "path": cache_path})
    except Exception as exc:
        print(f"[WARN] Falling back to live TTS for {tool_name}: {exc}")
        speech_request_queue.put({"kind": "text", "text": get_tool_prompt_text(tool_name), "silence_ms": 0})

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
    """Transcribe audio data using local Whisper or OpenAI API"""
    try:
        if USE_CLOUD_STT:
            # Use OpenAI Whisper API
            print("[STT] Using OpenAI Whisper API...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio_data, RATE)
            
            try:
                with open(tmp_path, 'rb') as audio_file:
                    transcript = openai_client.audio.transcriptions.create(
                        model=OPENAI_STT_MODEL,
                        file=audio_file,
                        language=STT_LANGUAGE
                    )
                text = transcript.text.strip()
                return text
            finally:
                os.remove(tmp_path)
        else:
            # Use local faster-whisper
            segments, info = whisper_model.transcribe(
                audio_data.astype(np.float32) / 32768.0,
                language=STT_LANGUAGE, 
                beam_size=1,
                without_timestamps=True,
                best_of=1,
                vad_filter=True
            )
            text = "".join([seg.text for seg in segments]).strip()
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
# TTS (Piper) Logic
# =============================

def ensure_piper_model():
    """Download and load the Piper TTS voice model"""
    global piper_voice
    
    if os.path.isfile(PIPER_VOICE_DEFAULT) and piper_voice is not None:
        return piper_voice
    
    print("[TTS] Ensuring Piper voice model...")
    os.makedirs(PIPER_DIR, exist_ok=True)
    
    # Download from HuggingFace if not present
    if not os.path.isfile(PIPER_VOICE_DEFAULT):
        print(f"[TTS] Downloading Piper voice model...")
        voice_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
        try:
            subprocess.run(["wget", "-c", voice_url, "-O", PIPER_VOICE_DEFAULT], check=True)
            print("[TTS] Piper voice model downloaded.")
        except Exception as e:
            print(f"[ERROR] Failed to download Piper model: {e}", file=sys.stderr)
            return None
    
    # Load the model
    try:
        print("[TTS] Loading Piper voice...")
        piper_voice = PiperVoice.load(PIPER_VOICE_DEFAULT)
        print("[TTS] Piper voice loaded successfully!")
        return piper_voice
    except Exception as e:
        print(f"[ERROR] Failed to load Piper voice: {e}", file=sys.stderr)
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


def cleanup_llm_server():
    """Stop the local LLM server if it is running."""
    global llm_process

    if not llm_process or llm_process.poll() is not None:
        return

    try:
        llm_process.terminate()
        llm_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        llm_process.kill()
        llm_process.wait(timeout=5)
    except Exception as e:
        print(f"[WARN] Failed to stop LLM server cleanly: {e}", file=sys.stderr)
    finally:
        llm_process = None


def handle_shutdown_signal(signum, frame):
    """Handle process termination from systemd or the shell."""
    print("\n[EXIT] Shutdown requested.")
    cleanup_llm_server()
    raise SystemExit(0)

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
        "You are a home assistant. Be concise for simple requests, but if the user asks for a detailed or long answer, provide a thorough response and keep track of recent conversation context."
    )
    
    if model_type == "qwen":
        # Qwen format: <|im_start|>role\ncontent<|im_end|>
        formatted = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{prompt} /no_think<|im_end|>\n<|im_start|>assistant\n"
    else:  # Phi format
        # Phi-4-mini-instruct format: <|role|>content<|end|>
        formatted = f"<|system|>{system_msg}<|end|><|user|>{prompt}<|end|><|assistant|>"
    
    return formatted

def speak(text, silence_ms=2000):
    """Queue speech so playback stays serialized while wake-word detection keeps running."""

    if not text.strip():
        return

    try:
        ensure_output_sink()
        ensure_speech_worker()
        speech_request_queue.put({
            "kind": "text",
            "text": text,
            "silence_ms": silence_ms,
        })
        print("[TTS] Queued interruptible playback...")

    except Exception as e:
        print(f"[ERROR] TTS failed: {e}", file=sys.stderr)


# =============================
# Tool Handler Functions
# =============================

def get_time():
    """Get current time"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

def get_weather(location: str) -> str:
    """Get weather for a location (stub implementation)"""
    # In a real implementation, you would call a weather API
    return f"Weather for {location}: Sunny, 22°C (This is a stub - integrate a weather API)"

def set_reminder(message: str, minutes_from_now: int = 5) -> str:
    """Set a reminder (stub implementation)"""
    # In a real implementation, you would schedule this
    return f"Reminder set: '{message}' in {minutes_from_now} minutes"

def control_device(device: str, action: str) -> str:
    """Control smart home device (stub implementation)"""
    # In a real implementation, you would control actual devices
    return f"Device '{device}' turned {action}"


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool based on its name and input"""
    if tool_name == "get_time":
        return get_time()
    elif tool_name == "get_weather":
        return get_weather(tool_input.get("location", "current location"))
    elif tool_name == "search_wikipedia":
        return search_wikipedia(
            tool_input.get("query", ""),
            tool_input.get("max_sentences", 5)
        )
    elif tool_name == "search_web":
        return search_web(
            tool_input.get("query", ""),
            tool_input.get("num_results", 3)
        )
    elif tool_name == "spotify_play":
        return spotify_play(
            tool_input.get("track_or_artist", ""),
            tool_input.get("context", "track")
        )
    elif tool_name == "set_reminder":
        return set_reminder(
            tool_input.get("message", ""),
            tool_input.get("minutes_from_now", 5)
        )
    elif tool_name == "control_device":
        return control_device(
            tool_input.get("device", ""),
            tool_input.get("action", "")
        )
    else:
        return f"Unknown tool: {tool_name}"


def clean_llm_output(text: str) -> str:
    # Remove <think>...</think> blocks (multiline safe)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove any remaining angle-bracket tokens like <think>, </think>, <s>, </s>
    text = re.sub(r"</?[^>]+>", "", text)

    # Strip markdown headings, bullets, blockquotes, emphasis and code fences for cleaner spoken output.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"```+|`", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.*?) (?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"(?<!_)_(?!_)(.*?)(?<!_)_(?!_)", r"\1", text)
    text = text.replace("#", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_system_prompt():
    return (
        "You are a helpful home assistant. The user often speaks Dutch, but may also speak English. "
        "Always respond in the same language as the user's message. Use tools proactively whenever they "
        "would improve accuracy or complete the user's request. Use search_wikipedia for durable background "
        "knowledge such as science, history, art, people, places, and concepts. Use search_web for "
        "contemporary or fast-changing information such as football scores, news, product availability, "
        "release dates, and recent events. Use spotify_play whenever the user wants music, a song, an "
        "artist, an album, or a playlist. Do not guess when a tool would give a better answer. You may "
        "combine tools when needed. Be concise for simple requests, but if the user asks for a detailed, "
        "expansive, story-like, or thorough answer, provide a long natural spoken response. Avoid markdown, "
        "headings, bullet lists, numbered lists, hashtags, tables, or other written formatting. Write as if "
        "you are speaking aloud to a person. Maintain continuity with the recent conversation history when "
        "the user refers back to earlier discussion."
    )


def get_conversation_messages():
    if CONVERSATION_MEMORY_TURNS <= 0:
        return []
    return conversation_history[-(CONVERSATION_MEMORY_TURNS * 2):]


def remember_turn(user_text: str, assistant_text: str):
    cleaned_response = clean_llm_output(assistant_text)
    if not user_text.strip() or not cleaned_response:
        return

    conversation_history.append({"role": "user", "content": user_text.strip()})
    conversation_history.append({"role": "assistant", "content": cleaned_response})

    max_messages = max(0, CONVERSATION_MEMORY_TURNS * 2)
    if max_messages and len(conversation_history) > max_messages:
        del conversation_history[:-max_messages]

def run_llm(prompt):
    """Run LLM using local llama.cpp or OpenAI API with tool support"""
    try:
        if USE_CLOUD_LLM:
            # Use OpenAI API with tool calling
            print("[LLM] Using OpenAI API with tools...")
            messages = [{"role": "system", "content": build_system_prompt()}]
            messages.extend(get_conversation_messages())
            messages.append({"role": "user", "content": prompt})
            
            # First API call with tools
            response = openai_client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                temperature=0.7,
                max_completion_tokens=LLM_MAX_TOKENS
            )
            
            # Check if the model wants to call tools
            if response.choices[0].message.tool_calls:
                # Process tool calls
                tool_calls = response.choices[0].message.tool_calls
                response_text = ""
                
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_input = json.loads(tool_call.function.arguments)
                    print(f"[TOOL] Calling {tool_name}({tool_input})... ", end="", flush=True)

                    enqueue_tool_prompt(tool_name)
                    
                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_input)
                    print(f"✓")
                    
                    # Add assistant message and tool result to conversation
                    messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
                
                # Second API call with tool results
                response = openai_client.chat.completions.create(
                    model=OPENAI_LLM_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_completion_tokens=LLM_MAX_TOKENS
                )
                
                response_text = response.choices[0].message.content
                print(response_text, end="", flush=True)
                cleaned_response = clean_llm_output(response_text)
                remember_turn(prompt, cleaned_response)
                return cleaned_response
            else:
                # No tool calls, just return the response
                response_text = response.choices[0].message.content
                print(response_text, end="", flush=True)
                cleaned_response = clean_llm_output(response_text)
                remember_turn(prompt, cleaned_response)
                return cleaned_response
        else:
            # Use local llama.cpp
            url = "http://127.0.0.1:8080/completion"
            prompt_with_history = prompt
            history_messages = get_conversation_messages()
            if history_messages:
                history_lines = []
                for message in history_messages:
                    role_name = "User" if message["role"] == "user" else "Assistant"
                    history_lines.append(f"{role_name}: {message['content']}")
                prompt_with_history = (
                    "Recent conversation:\n"
                    + "\n".join(history_lines)
                    + f"\nUser: {prompt}\nAssistant:"
                )
            formatted_prompt = format_prompt(prompt_with_history)
            
            data = json.dumps({
                "prompt": formatted_prompt,
                "n_predict": LLM_MAX_TOKENS,
                "temperature": 0.7,
                "stream": True
            }).encode("utf-8")
            
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
                cleaned_response = clean_llm_output(response_text)
                remember_turn(prompt, cleaned_response)
                return cleaned_response
            
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
    ensure_speech_worker()
    
    # Only download/check LLM if using local LLM
    if not USE_CLOUD_LLM:
        ensure_llm_safe()
    
    # Select output audio sink, preferring wired/USB and falling back to Bluetooth
    ensure_output_sink()
    
    # Load Piper TTS model only if using local TTS
    global piper_voice
    if not USE_CLOUD_TTS:
        piper_voice = ensure_piper_model()
        if not piper_voice:
            print("[ERROR] Could not load Piper TTS model!")
            return

    start_tool_prompt_cache_warmup()
    
    # Test speaker on startup
    print("[AUDIO] Testing speaker...")
    speak("Gooooodmorning! I'm ready to assist you. Just call my name to get my attention!")

    # Start LLM server only if using local LLM
    if not USE_CLOUD_LLM:
        if not start_llm_server():
            print("Could not start LLM server. Check paths!")
            return

    print("[INFO] Loading Vosk model for wake word detection...")
    stt_model = Model(VOSK_PATH)
    wake_recognizer = KaldiRecognizer(stt_model, RATE, '["hey homie", "[unk]"]')
    
    # Load Whisper model only if using local STT
    whisper_model = None
    if not USE_CLOUD_STT:
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
                        interrupted_speaking = is_speaking_active()
                        if interrupted_speaking:
                            print("[TTS] Interrupting current playback...")
                            stop_speaking()
                            flush_audio_queue()
                            rollback_buffer = []
                        state = "LISTENING"
                        play_feedback_beep(880, 120)

                        # 🔁 preload rollback audio
                        listening_audio = [] if interrupted_speaking else [f.copy() for f in rollback_buffer]
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
                    play_feedback_beep(660, 160)

                    if SAVE_DEBUG_RECORDINGS:
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
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    try:
        main()
    except KeyboardInterrupt:
        cleanup_llm_server()
        print("\n[EXIT] Goodbye.")