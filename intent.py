from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class Intent:
    type: str                  # e.g. "spotify", "chat", "search", "date"
    confidence: float = 1.0
    slots: Optional[Dict] = None
    raw_text: Optional[str] = None
