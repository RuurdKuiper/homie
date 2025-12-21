#!/usr/bin/env python3
"""Model downloading and management utilities."""

import os
import subprocess
import config

# Note: LLM_URL is not defined in config, so we'll need to handle this
# For now, we'll leave ensure_llm_safe as a placeholder that checks if the model exists


def ensure_llm_safe():
    """Ensure LLM model file exists (download if needed)"""
    if os.path.isfile(config.LLM_PATH):
        # Optional: check if file is too small (meaning it failed earlier)
        if os.path.getsize(config.LLM_PATH) > 1000000: # > 1MB
            return
            
    # Note: LLM_URL is commented out in config, so we can't download automatically
    # User should download manually or uncomment LLM_URL in config.py
    print(f"[INFO] LLM model not found at {config.LLM_PATH}")
    print("[INFO] Please download the model manually or uncomment LLM_URL in config.py")

