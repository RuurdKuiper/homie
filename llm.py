#!/usr/bin/env python3
"""LLM server management and client for llama.cpp."""

import json
import time
import subprocess
import urllib.request
import re
import config

# Global to handle cleanup
llm_process = None


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


def get_model_type():
    """Detect model type from LLM_NAME"""
    if "phi" in config.LLM_NAME.lower():
        return "phi"
    elif "qwen" in config.LLM_NAME.lower():
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


def clean_llm_output(text: str) -> str:
    """Clean LLM output by removing special tokens and normalizing whitespace"""
    # Remove <think>...</think> blocks (multiline safe)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove any remaining angle-bracket tokens like <think>, </think>, <s>, </s>
    text = re.sub(r"</?[^>]+>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def run_llm(prompt):
    """Send prompt to LLM server and return response"""
    url = "http://127.0.0.1:8080/completion"
    formatted_prompt = format_prompt(prompt)
    
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

