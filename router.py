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

    # Check for explicit long answer requests
    long_answer_keywords = [
        "tell me everything", "can you tell me", "what do you know", "look up",
        "explain", "describe", "what is", "who is", "what are", "can you explain",
        "tell me about", "what can you tell me", "everything about", "all about"
    ]
    
    if any(k in t for k in long_answer_keywords):
        return Intent(type="chat_long", raw_text=text)

    # Short answers for brief questions
    if len(t.split()) <= 6:
        return Intent(type="chat_short", raw_text=text)

    # --- fallback to short for medium-length questions ---
    return Intent(type="chat_short", raw_text=text)
