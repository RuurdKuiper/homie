#!/usr/bin/env python3
"""Long-form chat with Wikipedia context and larger LLM model."""

import json
import sys
import urllib.parse
import urllib.request
import llm
import config


def _fetch_wikipedia_summary(query: str, max_sentences: int = 3) -> str:
    """Fetch a summary from Wikipedia for the given query.
    
    Returns:
        Summary text, or empty string if not found
    """
    try:
        # Wikipedia API: get page summary
        # Format: https://en.wikipedia.org/api/rest_v1/page/summary/{title}
        # First, search for the page
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1
        }
        
        search_params = urllib.parse.urlencode(params)
        with urllib.request.urlopen(f"{search_url}?{search_params}", timeout=5) as response:
            search_data = json.loads(response.read().decode('utf-8'))
            
            if "query" in search_data and "search" in search_data["query"] and len(search_data["query"]["search"]) > 0:
                page_title = search_data["query"]["search"][0]["title"]
                
                # Get summary using REST API
                encoded_title = urllib.parse.quote(page_title)
                summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
                
                with urllib.request.urlopen(summary_url, timeout=5) as summary_response:
                    summary_data = json.loads(summary_response.read().decode('utf-8'))
                    extract = summary_data.get("extract", "")
                    
                    # Limit to first N sentences
                    sentences = extract.split('. ')
                    if len(sentences) > max_sentences:
                        extract = '. '.join(sentences[:max_sentences]) + '.'
                    
                    return extract
    except Exception as e:
        print(f"[WARNING] Wikipedia lookup failed: {e}", file=sys.stderr)
        return ""
    
    return ""


def _extract_keywords(text: str) -> str:
    """Extract potential Wikipedia search keywords from user query."""
    # Remove common question words and extract nouns/key phrases
    text_lower = text.lower()
    
    # Remove question words
    question_words = ["what", "is", "are", "can", "you", "tell", "me", "about", "do", "know", 
                     "everything", "look", "up", "explain", "describe", "who", "when", "where", "why", "how"]
    
    words = text.split()
    keywords = [w for w in words if w.lower() not in question_words and len(w) > 2]
    
    # Take first 2-3 meaningful words
    if len(keywords) >= 2:
        return ' '.join(keywords[:3])
    elif len(keywords) == 1:
        return keywords[0]
    else:
        # Fallback: use original text
        return text


def handle(text: str) -> str:
    """Handle long-form chat request with Wikipedia context."""
    # Extract keywords for Wikipedia search
    keywords = _extract_keywords(text)
    
    # Fetch Wikipedia context
    wiki_context = ""
    if keywords:
        wiki_context = _fetch_wikipedia_summary(keywords, max_sentences=3)
    
    # Build prompt with context
    if wiki_context:
        prompt = f"Based on this information: {wiki_context}\n\nUser question: {text}\n\nProvide a detailed, informative answer."
    else:
        prompt = f"User question: {text}\n\nProvide a detailed, informative answer based on your knowledge."
    
    # Use the larger LLM model for long answers
    response = llm.run_llm_long(prompt)
    
    return response if response else "I'm sorry, I couldn't generate a response to that question."
