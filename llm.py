#!/usr/bin/env python3
"""LLM server management and client for llama.cpp."""

import json
import sys
import time
import subprocess
import urllib.request
import re
import config

# Global to handle cleanup
llm_process = None
llm_long_process = None


def start_llm_server():
    """Start the llama.cpp server process"""
    global llm_process
    print("[LLM] 🚀 Preloading model into RAM...")
    
    cmd = [
        config.LLAMA_SERVER_BIN,
        "-m", config.LLM_PATH,
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
    for _ in range(60): # Give it 60 seconds to load
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


def get_model_type(model_name=None):
    """Detect model type from model name"""
    if model_name is None:
        model_name = config.LLM_NAME
    
    if "phi" in model_name.lower():
        return "phi"
    elif "qwen" in model_name.lower():
        return "qwen"
    else:
        return "qwen"  # Default to Qwen format


def format_prompt(prompt, is_long=False, model_name=None):
    """Format prompt based on model type"""
    if is_long and model_name is None:
        model_name = config.LLM_LONG_NAME
    
    model_type = get_model_type(model_name)
    
    if is_long:
        system_msg = (
            "You are a helpful home assistant. Provide detailed, informative answers based on the context provided."
        )
    else:
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


def clean_llm_output(text: str) -> str:
    """Clean LLM output by removing special tokens and normalizing whitespace"""
    # Remove <think>...</think> blocks (multiline safe)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove any remaining angle-bracket tokens like <think>, </think>, <s>, </s>
    text = re.sub(r"</?[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def start_llm_long_server():
    """Start the larger LLM server for long answers"""
    global llm_long_process
    if llm_long_process is not None:
        # Check if still running
        if llm_long_process.poll() is None:
            return True  # Already running
    
    print("[LLM-LONG] 🚀 Starting large model server...")
    
    cmd = [
        config.LLAMA_SERVER_BIN,
        "-m", config.LLM_LONG_PATH,
        "--port", str(config.LLM_LONG_PORT),
        "--n-gpu-layers", "0", # Pi uses CPU
        "--threads", "4",
        "--ctx-size", "2048",  # Larger context for detailed answers
        "--log-disable"
    ]
    
    # Launch as background process
    llm_long_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to wake up
    print("[LLM-LONG] Waiting for server to be ready...", end="", flush=True)
    for _ in range(120):  # Give it more time (2 minutes) for larger model
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{config.LLM_LONG_PORT}/health") as f:
                if f.getcode() == 200:
                    print(" READY! ✅")
                    return True
        except:
            print(".", end="", flush=True)
            time.sleep(1)
    print(" TIMEOUT ❌")
    return False


def run_llm(prompt):
    """Send prompt to LLM server and return response"""
    url = "http://127.0.0.1:8080/completion"
    formatted_prompt = format_prompt(prompt, is_long=False)
    
    data = json.dumps({
        "prompt": formatted_prompt,
        "n_predict": config.LLM_MAX_TOKENS,
        "temperature": config.LLM_TEMPERATURE,
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


def run_llm_long(prompt):
    """Send prompt to large LLM server and return response"""
    # Ensure large model server is running
    if not start_llm_long_server():
        print("[ERROR] Large LLM server not available, falling back to small model", file=sys.stderr)
        return run_llm(prompt)  # Fallback to small model
    
    url = f"http://127.0.0.1:{config.LLM_LONG_PORT}/completion"
    formatted_prompt = format_prompt(prompt, is_long=True, model_name=config.LLM_LONG_NAME)
    
    data = json.dumps({
        "prompt": formatted_prompt,
        "n_predict": config.LLM_LONG_MAX_TOKENS,
        "temperature": config.LLM_LONG_TEMPERATURE,
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

