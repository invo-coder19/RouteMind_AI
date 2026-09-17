"""
classifier/features.py
======================
Feature extraction for the LLM prompt complexity classifier.

All features are derived purely from the raw prompt text — no LLM calls,
no external resources.  The public API is a single function:

    extract_features(prompt: str) -> dict[str, float | int]

The returned dict is stable (same keys, same order) across calls and is
designed to be fed directly into a scikit-learn pipeline via
``DictVectorizer`` or ``pd.DataFrame([features])``.
"""
from __future__ import annotations

import re
import string
from typing import Final

# ---------------------------------------------------------------------------
# Keyword lists — complexity signals
# ---------------------------------------------------------------------------

#: Keywords strongly associated with complex prompts (code gen, multi-step
#: reasoning, design tasks).
_COMPLEX_KEYWORDS: Final[tuple[str, ...]] = (
    "write a",
    "implement",
    "design a",
    "build a",
    "create a",
    "architect",
    "debug",
    "optimize",
    "refactor",
    "step by step",
    "step-by-step",
    "algorithm",
    "from scratch",
    "how would you",
    "walk through",
    "trade-off",
    "trade off",
    "pros and cons",
    "compare and contrast",
    "in detail",
    "end-to-end",
    "production",
    "at scale",
    "distributed",
    "microservice",
    "recursion",
    "recursive",
    "concurrently",
    "multi-step",
    "systems design",
)

#: Keywords associated with moderate complexity (explanation, comparison,
#: summarisation).
_MODERATE_KEYWORDS: Final[tuple[str, ...]] = (
    "explain",
    "describe",
    "compare",
    "summarize",
    "summarise",
    "difference between",
    "how does",
    "how do",
    "why is",
    "what are the",
    "advantages",
    "disadvantages",
    "overview",
    "when to use",
    "when would",
    "example",
)

#: Keywords strongly associated with simple prompts (factual look-up).
_SIMPLE_KEYWORDS: Final[tuple[str, ...]] = (
    "what is",
    "who is",
    "who was",
    "when did",
    "when was",
    "define",
    "definition of",
    "capital of",
    "how many",
    "convert",
    "translate",
    "list the",
    "name the",
    "spell",
)

# ---------------------------------------------------------------------------
# Question-type heuristics
# ---------------------------------------------------------------------------

_SIMPLE_STARTERS: Final[tuple[str, ...]] = ("what is", "who is", "who was",
                                             "when did", "when was", "where is",
                                             "define", "how many", "how much")

_MODERATE_STARTERS: Final[tuple[str, ...]] = ("how does", "how do", "why is",
                                               "why are", "what are", "explain",
                                               "describe", "compare", "summarize",
                                               "summarise")

_COMPLEX_STARTERS: Final[tuple[str, ...]] = ("write a", "implement", "design",
                                              "build", "create a function",
                                              "create a class", "how would you",
                                              "debug", "optimize", "walk through",
                                              "architect")


def _normalise(text: str) -> str:
    """Lowercase and strip leading/trailing whitespace."""
    return text.lower().strip()


def _word_count(text: str) -> int:
    """Number of whitespace-separated tokens."""
    return len(text.split())


def _char_count(text: str) -> int:
    """Total character count (including spaces)."""
    return len(text)


def _count_keywords(text_lower: str, keywords: tuple[str, ...]) -> int:
    """Count how many keyword phrases appear in *text_lower*."""
    return sum(1 for kw in keywords if kw in text_lower)


def _has_code_block(text: str) -> int:
    """Return 1 if the prompt contains a fenced code block (```), else 0."""
    return int("```" in text)


def _has_url(text: str) -> int:
    """Return 1 if the prompt contains a URL, else 0."""
    return int(bool(re.search(r"https?://\S+", text)))


def _question_mark_count(text: str) -> int:
    """Number of question marks — multiple Qs often → moderate/complex."""
    return text.count("?")


def _sentence_count(text: str) -> int:
    """Approximate number of sentences (split on . ! ?)."""
    return max(1, len(re.split(r"[.!?]+", text.strip())))


def _starts_with_simple(text_lower: str) -> int:
    return int(any(text_lower.startswith(s) for s in _SIMPLE_STARTERS))


def _starts_with_moderate(text_lower: str) -> int:
    return int(any(text_lower.startswith(s) for s in _MODERATE_STARTERS))


def _starts_with_complex(text_lower: str) -> int:
    return int(any(text_lower.startswith(s) for s in _COMPLEX_STARTERS))


def _avg_word_length(text: str) -> float:
    """Average length of words (punctuation stripped) — proxy for vocabulary complexity."""
    words = text.translate(str.maketrans("", "", string.punctuation)).split()
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def _contains_code_tokens(text_lower: str) -> int:
    """Heuristic: does the prompt contain tokens typical in code contexts."""
    code_tokens = ("def ", "class ", "import ", "return ", "for ", "while ",
                   "if ", "else:", "elif ", "()", "->", "=>", "::", "[]", "{}")
    return int(any(tok in text_lower for tok in code_tokens))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(prompt: str) -> dict[str, float | int]:
    """Extract a fixed-size feature dict from a raw prompt string.

    Parameters
    ----------
    prompt:
        The raw prompt text (UTF-8 string, any length).

    Returns
    -------
    dict[str, float | int]
        A flat dictionary with stable keys suitable for use with
        ``sklearn.feature_extraction.DictVectorizer`` or direct conversion to
        a ``pandas.DataFrame``.  All values are numeric (int or float).

    Examples
    --------
    >>> feats = extract_features("What is the capital of France?")
    >>> feats["word_count"]
    7
    >>> feats["simple_keyword_count"] >= 1
    True
    """
    low = _normalise(prompt)

    return {
        # --- Length features ------------------------------------------------
        "word_count": _word_count(prompt),
        "char_count": _char_count(prompt),
        "sentence_count": _sentence_count(prompt),
        "avg_word_length": round(_avg_word_length(prompt), 4),
        "question_mark_count": _question_mark_count(prompt),
        # --- Keyword complexity signals -------------------------------------
        "complex_keyword_count": _count_keywords(low, _COMPLEX_KEYWORDS),
        "moderate_keyword_count": _count_keywords(low, _MODERATE_KEYWORDS),
        "simple_keyword_count": _count_keywords(low, _SIMPLE_KEYWORDS),
        # --- Question type heuristics ---------------------------------------
        "starts_with_simple": _starts_with_simple(low),
        "starts_with_moderate": _starts_with_moderate(low),
        "starts_with_complex": _starts_with_complex(low),
        # --- Content signals -----------------------------------------------
        "has_code_block": _has_code_block(prompt),
        "has_url": _has_url(prompt),
        "contains_code_tokens": _contains_code_tokens(low),
    }
