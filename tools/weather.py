#!/usr/bin/env python3
"""Weather information using wttr.in API."""

import json
import sys
import re
import urllib.parse
import urllib.request
import config

# Default location from config
DEFAULT_LOCATION = config.WEATHER_DEFAULT_LOCATION


def _fetch_weather(location: str) -> dict:
    """Fetch weather data from wttr.in API."""
    try:
        # wttr.in API: format=j1 returns JSON format
        # URL encode the location to handle special characters and spaces
        encoded_location = urllib.parse.quote(location, safe='')
        url = f"https://wttr.in/{encoded_location}?format=j1"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception as e:
        print(f"[ERROR] Weather API request failed for '{location}': {e}", file=sys.stderr)
        return None


def _extract_location(data: dict) -> str:
    """Extract location name from weather data."""
    if data and "nearest_area" in data and len(data["nearest_area"]) > 0:
        area = data["nearest_area"][0]
        if "areaName" in area and len(area["areaName"]) > 0:
            return area["areaName"][0]["value"]
    return "Unknown location"


def _is_valid_weather_data(data: dict) -> bool:
    """Check if weather data is valid."""
    if not data:
        return False
    if "current_condition" not in data or len(data["current_condition"]) == 0:
        return False
    current = data["current_condition"][0]
    # Check if we have essential data
    if "temp_C" not in current or current.get("temp_C") == "N/A":
        return False
    return True


def _format_weather_response(data: dict) -> str:
    """Format weather data into a natural language response."""
    if not _is_valid_weather_data(data):
        return None  # Return None to indicate invalid data
    
    current = data["current_condition"][0]
    temp_c = current.get("temp_C", "N/A")
    condition = current.get("weatherDesc", [{}])[0].get("value", "unknown conditions")
    feels_like = current.get("FeelsLikeC", "N/A")
    humidity = current.get("humidity", "N/A")
    
    location_name = _extract_location(data)
    
    # Format a natural response
    response = f"The weather in {location_name} is {condition.lower()}, {temp_c} degrees Celsius"
    if feels_like != "N/A" and feels_like != temp_c:
        response += f", feels like {feels_like} degrees"
    response += f", with {humidity} percent humidity."
    
    return response


def handle(location: str = None) -> str:
    """Handle weather request. If location is None, uses default location."""
    if location is None:
        location = DEFAULT_LOCATION
    
    data = _fetch_weather(location)
    if data is None:
        return None  # Return None to indicate failure
    
    response = _format_weather_response(data)
    if response is None:
        return None  # Invalid data
    
    return response


def _extract_location_from_text(text: str) -> str:
    """Extract location name from user text using simple patterns."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Common phrases that should NOT be extracted as locations
    invalid_phrases = [
        'like today', 'like tomorrow', 'like right now', 'like now',
        'like this', 'like that', 'like here', 'like there',
        'today', 'tomorrow', 'right now', 'now', 'here', 'there',
        'this week', 'next week', 'this month', 'next month'
    ]
    
    # Patterns to extract location: "weather in X", "temperature in X", "weather at X", etc.
    # Handle both "weather in X" and "weather like in X" patterns
    patterns = [
        r'weather like in (.+?)(?:\?|$|\.)',  # "weather like in Amsterdam"
        r'temperature like in (.+?)(?:\?|$|\.)',  # "temperature like in Paris"
        r'weather in (.+?)(?:\?|$|\.)',  # "weather in Amsterdam"
        r'temperature in (.+?)(?:\?|$|\.)',  # "temperature in Paris"
        r'weather at (.+?)(?:\?|$|\.)',  # "weather at London"
        r'temperature at (.+?)(?:\?|$|\.)',  # "temperature at London"
        r'how.*weather.*in (.+?)(?:\?|$|\.)',  # "how is the weather in X"
        r'what.*weather.*in (.+?)(?:\?|$|\.)',  # "what is the weather in X"
        r'what.*weather like in (.+?)(?:\?|$|\.)',  # "what is the weather like in X"
        # Only match "weather X" if X doesn't start with "like"
        r'weather (?!like)(.+?)(?:\?|$|\.)',
        r'temperature (?!like)(.+?)(?:\?|$|\.)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            
            # Check if extracted location is an invalid phrase
            if any(invalid in location.lower() for invalid in invalid_phrases):
                continue  # Skip this match
            
            # Remove trailing filler words only (preserve location names like "The Hague")
            location = re.sub(r'\b(like|for|a|an|of|is|will|does|what|how|tell|me|about)\b\s*$', '', location, flags=re.IGNORECASE)
            location = location.strip(' ,.')
            
            # Additional validation: if location contains common non-location words, skip
            if any(word in location.lower() for word in ['today', 'tomorrow', 'now', 'here', 'there', 'this', 'that']):
                continue
            
            if location and len(location) > 1:
                return location
    
    # If no pattern matched, try to find capitalized words (likely location names)
    # Look for patterns like "weather Paris" or "weather New York"
    words = text.split()
    weather_keywords = ['weather', 'temperature', 'rain', 'temp']
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in weather_keywords and i + 1 < len(words):
            # Check if next word is "like" - if so, skip
            if i + 1 < len(words) and words[i + 1].lower() == 'like':
                continue
            
            # Take next 1-3 words as potential location
            potential_location = ' '.join(words[i+1:i+4])
            # Remove trailing punctuation
            potential_location = re.sub(r'[.,!?]+$', '', potential_location)
            
            # Check if it's an invalid phrase
            if any(invalid in potential_location.lower() for invalid in invalid_phrases):
                continue
            
            if potential_location and len(potential_location) > 1:
                return potential_location.strip()
    
    return None


def _normalize_location(location: str) -> str:
    """Normalize location string for wttr.in API."""
    if not location:
        return location
    
    # Clean up the location string
    location = location.strip()
    
    # Handle common location name variations
    location = re.sub(r'\s+', ' ', location)  # Multiple spaces to single space
    
    # URL encode the location for the API
    # wttr.in can handle most location formats, so we'll pass it as-is
    return location


def handle_with_text(text: str) -> str:
    """Handle weather request, extracting location from text if mentioned.
    
    If the requested location cannot be found or fails, falls back to default location (Utrecht).
    """
    location = None
    requested_location = None
    
    if text:
        # Extract location from text
        extracted = _extract_location_from_text(text)
        if extracted:
            requested_location = extracted  # Keep original for error message
            location = _normalize_location(extracted)
    
    # Try requested location first (if specified)
    if location and location != DEFAULT_LOCATION:
        response = handle(location)
        if response is not None:
            return response
        # If failed, log and fall back to default
        print(f"[WARNING] Weather request for '{requested_location}' failed, falling back to {DEFAULT_LOCATION}", file=sys.stderr)
    
    # Fall back to default location
    response = handle(DEFAULT_LOCATION)
    if response is None:
        return "Sorry, I couldn't fetch the weather information."
    
    # If we fell back, mention it in the response
    if location and location != DEFAULT_LOCATION:
        return f"I couldn't find weather for that location. {response}"
    
    return response
