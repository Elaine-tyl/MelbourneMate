"""English token helpers for the collection quality analysis."""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    # Keep domain words such as "visa", "cover", and "bond".
    [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do",
        "does", "for", "from", "has", "have", "how", "i", "if", "in", "is",
        "it", "its", "me", "my", "of", "on", "or", "that", "the", "their",
        "there", "they", "this", "to", "was", "what", "when", "where",
        "which", "who", "will", "with", "you", "your",
    ]
)

def terms(text: str) -> set[str]:
    """Return lower-case English content words for overlap analysis."""
    lowered = text.lower()
    return {t for t in _WORD.findall(lowered) if t not in STOPWORDS and len(t) > 1}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment(query: set[str], document: set[str]) -> float:
    """Return the share of query terms also found in the document."""
    if not query:
        return 0.0
    return len(query & document) / len(query)
