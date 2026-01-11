#!/usr/bin/env python3
"""Acknowledgment messages for immediate feedback during processing."""

import random

# Acknowledgment messages for weather requests
WEATHER_ACKNOWLEDGMENTS = [
    "Let me check the forecast for you, just one second...",
    "Checking the weather now...",
    "Let me look up the weather for you...",
    "One moment, I'll get the weather information...",
    "Checking the forecast, hold on...",
    "Let me fetch the weather data...",
    "Looking up the weather for you...",
]

# Acknowledgment messages for longer LLM queries (chat_long)
THINKING_ACKNOWLEDGMENTS = [
    "Interesting question, let me think about that for a second...",
    "That's a good one, give me a moment...",
    "Let me think about that...",
    "Hmm, let me consider that...",
    "That's interesting, let me ponder that...",
    "Give me a second to think about that...",
    "Let me work through that for you...",
]


def get_acknowledgment(intent_type: str) -> str:
    """Get a random acknowledgment message for the given intent type.
    
    Args:
        intent_type: The type of intent (e.g., "weather", "chat_long")
    
    Returns:
        A random acknowledgment message, or empty string if no acknowledgment needed
    """
    if intent_type == "weather":
        return random.choice(WEATHER_ACKNOWLEDGMENTS)
    elif intent_type == "chat_long":
        return random.choice(THINKING_ACKNOWLEDGMENTS)
    else:
        return ""  # No acknowledgment for other intent types
