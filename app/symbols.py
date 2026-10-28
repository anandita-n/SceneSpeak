"""Phase 3: symbol mapping via ARASAAC.

Real AAC practice uses established, clinically-vetted symbol sets rather
than freshly generated illustrations — consistency of symbols across a
child's boards matters more than novelty. ARASAAC (https://arasaac.org) is
a free, openly-licensed AAC pictogram set with a public search API, so we
map each generated word to a real symbol instead of inventing new art.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import requests
from pydantic import BaseModel

_MAX_WORKERS = 10

ARASAAC_SEARCH_URL = "https://api.arasaac.org/api/pictograms/en/search/{query}"
ARASAAC_IMAGE_URL = "https://static.arasaac.org/pictograms/{symbol_id}/{symbol_id}_500.png"
_TIMEOUT = 5

# Lookups are also cached on disk, so a restart (or a fresh container with a
# persistent volume) doesn't repeat the network round-trips.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scenespeak.db"
_NETWORK_ERROR = object()   # a failed request: retry next time, never cache
_NOT_CACHED = object()


class SymbolEntry(BaseModel):
    word: str
    symbol_id: int | None = None
    image_url: str | None = None
    # Set by the verifier (app/verification.py): None = not scored.
    score: float | None = None
    low_confidence: bool = False


def _relevance(term_lower: str, result: dict) -> tuple[int, int] | None:
    """Lower is better. None means "not actually relevant" — ARASAAC's search
    is loose full-text matching, not ranked by relevance, so a raw top result
    can be a near-random unrelated pictogram (observed: searching "all done"
    returned a "garden" symbol as its top AAC-flagged hit). We only trust a
    result if the search term and one of its keywords actually overlap."""
    keywords = [k["keyword"].lower() for k in result.get("keywords", [])]
    if term_lower in keywords:
        overlap_score = 0
    elif any(term_lower in kw or kw in term_lower for kw in keywords):
        overlap_score = 1
    else:
        return None
    aac_bonus = 0 if result.get("aac") else 1
    return (overlap_score, aac_bonus)


def _search(term: str):
    """The best relevant symbol id for `term`, None if ARASAAC has no relevant
    symbol, or _NETWORK_ERROR if the request failed."""
    try:
        resp = requests.get(ARASAAC_SEARCH_URL.format(query=term), timeout=_TIMEOUT)
        if resp.status_code == 404:  # ARASAAC answers "no results" with a 404
            return None
        resp.raise_for_status()
    except requests.RequestException:
        return _NETWORK_ERROR

    results = resp.json()
    if not results:
        return None

    term_lower = term.lower()
    scored = [(_relevance(term_lower, r), r) for r in results]
    scored = [(score, r) for score, r in scored if score is not None]
    if not scored:
        return None

    scored.sort(key=lambda pair: pair[0])
    return scored[0][1].get("_id")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("CREATE TABLE IF NOT EXISTS symbol_cache (word TEXT PRIMARY KEY, symbol_id INTEGER)")
    return conn


def _cache_get(word: str):
    with closing(_connect()) as conn:
        row = conn.execute("SELECT symbol_id FROM symbol_cache WHERE word = ?", (word.lower(),)).fetchone()
    return _NOT_CACHED if row is None else row[0]


def _cache_put(word: str, symbol_id: int | None) -> None:
    with closing(_connect()) as conn:
        conn.execute("INSERT OR REPLACE INTO symbol_cache (word, symbol_id) VALUES (?, ?)", (word.lower(), symbol_id))
        conn.commit()


def _lookup(word: str):
    """Network lookup with the multi-word fallback (first word of a phrase)."""
    symbol_id = _search(word)
    if symbol_id is None and " " in word:
        symbol_id = _search(word.split(" ")[0])
    return symbol_id


_memory: dict[str, SymbolEntry] = {}  # answered lookups only — failures aren't remembered


def find_symbol(word: str) -> SymbolEntry:
    """Look up a real AAC symbol for `word` (memory cache, then disk cache,
    then ARASAAC). Falls back to the first word of a multi-word phrase if the
    exact phrase has no match, and returns a symbol-less entry (word only) if
    nothing is found at all — the board still shows the word as text rather
    than silently dropping it."""
    if word in _memory:
        return _memory[word]
    symbol_id = _cache_get(word)
    if symbol_id is _NOT_CACHED:
        symbol_id = _lookup(word)
        if symbol_id is _NETWORK_ERROR:
            return SymbolEntry(word=word)  # show as text now; retried on the next board
        _cache_put(word, symbol_id)

    entry = (SymbolEntry(word=word) if symbol_id is None else
             SymbolEntry(word=word, symbol_id=symbol_id, image_url=ARASAAC_IMAGE_URL.format(symbol_id=symbol_id)))
    _memory[word] = entry
    return entry


def clear_memory_cache() -> None:
    """Forget in-memory lookups (the disk cache is kept)."""
    _memory.clear()


def map_words_to_symbols(words: list[str]) -> list[SymbolEntry]:
    """Look up symbols for every word concurrently. This was the actual
    bottleneck in board generation — each word is an independent network
    round-trip to ARASAAC's API, and doing them one at a time measured at
    over 1 second per word (35+ seconds for a realistic 32-word board),
    dwarfing every model-inference step in the pipeline combined. These
    lookups don't depend on each other, so a thread pool (I/O-bound work,
    not CPU-bound, so the GIL isn't a limiting factor here) lets them
    happen in parallel instead of queued one after another."""
    if not words:
        return []
    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(words))) as executor:
        return list(executor.map(find_symbol, words))
