from __future__ import annotations
import re

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace; keep newlines (notes are section-ish)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove excessive spaces/tabs
    text = re.sub(r"[\t ]{2,}", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
