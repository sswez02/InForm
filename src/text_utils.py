import re
from typing import List

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "in",
    "on",
    "of",
    "for",
    "with",
    "to",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "it",
    "as",
    "by",
    "from",
    "at",
    "be",
    "we",
    "our",
    "their",
}


def normalise(text: str) -> str:
    text = text.lower()
    # Remove non-alphanumeric (except spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenise(text: str) -> List[str]:
    norm = normalise(text)
    tokens = norm.split(" ")
    return [t for t in tokens if t and t not in STOPWORDS]
