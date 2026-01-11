import re
from intent import Intent

def route(text: str) -> Intent:
    t = text.lower()

    # --- fast rules (cheap & reliable) ---
    if any(k in t for k in ["pause", "resume", "next song", "previous song"]):
        return Intent(type="spotify_control", raw_text=text)

    if any(k in t for k in ["play", "put on", "listen to"]):
        return Intent(type="spotify_play", raw_text=text)

    if any(k in t for k in ["weather", "temperature", "rain"]):
        return Intent(type="weather", raw_text=text)

    if any(k in t for k in ["what day", "what date", "today"]):
        return Intent(type="date")

    if len(t.split()) <= 6:
        return Intent(type="chat_short", raw_text=text)

    # --- fallback ---
    return Intent(type="chat_long", raw_text=text)
