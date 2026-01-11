#!/usr/bin/env python3
"""Weather information using wttr.in API."""

import json
import sys
import re
import urllib.parse
import urllib.request

# Default location
DEFAULT_LOCATION = "Utrecht,Netherlands"


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


def _format_weather_response(data: dict) -> str:
    """Format weather data into a natural language response."""
    if not data or "current_condition" not in data or len(data["current_condition"]) == 0:
        return "Sorry, I couldn't fetch the weather information."
    
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
        return "Sorry, I couldn't fetch the weather information."
    return _format_weather_response(data)


def _extract_location_from_text(text: str) -> str:
    """Extract location name from user text using simple patterns."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    # Patterns to extract location: "weather in X", "temperature in X", "weather at X", etc.
    patterns = [
        r'weather in (.+?)(?:\?|$|\.)',
        r'temperature in (.+?)(?:\?|$|\.)',
        r'weather at (.+?)(?:\?|$|\.)',
        r'temperature at (.+?)(?:\?|$|\.)',
        r'weather (.+?)(?:\?|$|\.)',
        r'how.*weather.*in (.+?)(?:\?|$|\.)',
        r'what.*weather.*in (.+?)(?:\?|$|\.)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            location = match.group(1).strip()
            # Remove trailing filler words only (preserve location names like "The Hague")
            location = re.sub(r'\b(like|for|a|an|of|is|will|does|what|how|tell|me|about)\b\s*$', '', location, flags=re.IGNORECASE)
            location = location.strip(' ,.')
            if location and len(location) > 1:
                return location
    
    # If no pattern matched, try to find capitalized words (likely location names)
    # Look for patterns like "weather Paris" or "weather New York"
    words = text.split()
    weather_keywords = ['weather', 'temperature', 'rain', 'temp']
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in weather_keywords and i + 1 < len(words):
            # Take next 1-3 words as potential location
            potential_location = ' '.join(words[i+1:i+4])
            # Remove trailing punctuation
            potential_location = re.sub(r'[.,!?]+$', '', potential_location)
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
    """Handle weather request, extracting location from text if mentioned."""
    location = None
    
    if text:
        # Extract location from text
        extracted = _extract_location_from_text(text)
        if extracted:
            location = _normalize_location(extracted)
    
    # Default to Utrecht if no location found
    if not location:
        location = DEFAULT_LOCATION
    
    return handle(location)
