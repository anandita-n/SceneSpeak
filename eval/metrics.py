"""Metric functions for the SceneSpeak evaluation (pure, no I/O — easy to test).

Words are compared through `forms()`: lower-case, singular/base form via
WordNet's lemmatizer, plus a small alias table for the obvious synonyms a
board uses ("fridge" for "refrigerator", "kid" for "person"). WordNet
synonym *sets* are deliberately not used here: they are too loose for
scoring (e.g. "ball" would match "globe").
"""

from __future__ import annotations

import re
from functools import lru_cache
from statistics import mean

# Alias groups: any word in a group matches any other.
ALIAS_GROUPS: list[set[str]] = [
    {"person", "people", "man", "woman", "boy", "girl", "child", "kid", "baby", "adult", "player", "friend"},
    {"refrigerator", "fridge"},
    {"tv", "television", "screen"},
    {"couch", "sofa"},
    {"table", "dining table", "desk"},
    {"phone", "cell phone", "mobile phone", "cellphone"},
    {"ball", "sports ball", "football", "soccer ball", "basketball", "tennis ball", "baseball"},
    {"bicycle", "bike"},
    {"motorcycle", "motorbike"},
    {"plant", "potted plant", "flower pot"},
    {"cup", "mug"},
    {"glass", "wine glass"},
    {"glove", "baseball glove", "mitt"},
    {"bat", "baseball bat"},
    {"racket", "tennis racket", "racquet"},
    {"teddy", "teddy bear", "stuffed animal", "toy bear"},
    {"hair dryer", "hair drier", "hairdryer"},
    {"oven", "stove", "cooker"},
    {"sink", "basin"},
    {"toilet", "loo"},
    {"bag", "handbag", "backpack", "purse"},
    {"doughnut", "donut"},
    {"puppy", "dog"},
    {"kitten", "cat"},
    {"car", "automobile"},
]


@lru_cache(maxsize=1)
def _lemmatizer():
    from app.verification import _wordnet  # reuses the download-on-first-use setup

    return _wordnet()[1]


def normalize(word: str) -> str:
    """Lower-case, strip punctuation, and lemmatize each token ("swings" -> "swing")."""
    tokens = re.findall(r"[a-z]+", word.lower())
    lem = _lemmatizer()
    return " ".join(lem.lemmatize(t, "n") for t in tokens)


@lru_cache(maxsize=4096)
def forms(word: str) -> frozenset[str]:
    """Every form a word may match: its normalised form plus its alias group."""
    base = normalize(word)
    out = {base, word.lower().strip()}
    for group in ALIAS_GROUPS:
        normalised_group = {normalize(g) for g in group}
        if base in normalised_group:
            out |= normalised_group
    return frozenset(out)


def matches(word: str, labels: list[str]) -> bool:
    """True if `word` refers to any label in `labels`."""
    word_forms = forms(word)
    return any(word_forms & forms(label) for label in labels)


def score_board(scene_words: list[str], expected: list[str], not_present: list[str],
                ignore: list[str] | None = None) -> dict:
    """Score one board's scene-specific words against an image's labels.

    - precision: share of words that are in `expected`
    - hard_hallucination: share of words that are in `not_present` (definitely wrong)
    - not_in_expected: 1 - precision — the brief's broader "hallucination"
      measure; an upper bound, since labels never list every visible object
    - unlisted: share in neither list (can't be judged either way)
    - recall: share of `expected` labels that some word covers
    """
    ignore = ignore or []
    words = list(dict.fromkeys(w for w in scene_words if w.strip()))
    if not words:
        return {"n_words": 0, "precision": None, "hard_hallucination": None,
                "not_in_expected": None, "unlisted": None, "recall": 0.0 if expected else None}
    in_expected = [w for w in words if matches(w, expected)]
    in_absent = [w for w in words if not matches(w, expected) and matches(w, not_present)]
    unlisted = [w for w in words if not matches(w, expected) and not matches(w, not_present) and not matches(w, ignore)]
    covered = [e for e in expected if any(matches(w, [e]) for w in words)]
    n = len(words)
    return {
        "n_words": n,
        "precision": len(in_expected) / n,
        "hard_hallucination": len(in_absent) / n,
        "not_in_expected": 1 - len(in_expected) / n,
        "unlisted": len(unlisted) / n,
        "recall": len(covered) / len(expected) if expected else None,
    }


def grounding_rate(all_words: list[str], corpus_words: set[str]) -> float | None:
    """Share of suggested words (all categories) that appear in the AAC corpus."""
    words = [w for w in all_words if w.strip()]
    if not words:
        return None
    corpus = {normalize(c) for c in corpus_words}
    return sum(normalize(w) in corpus for w in words) / len(words)


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile (pct in 0-100)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(pct / 100 * len(ordered) + 0.4999))
    return ordered[min(rank, len(ordered)) - 1]


def mean_of(rows: list[dict], key: str) -> float | None:
    """Mean of a metric across rows, skipping rows where it's undefined."""
    vals = [r[key] for r in rows if r.get(key) is not None]
    return mean(vals) if vals else None
