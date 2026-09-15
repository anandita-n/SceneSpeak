"""Phase 3: symbol mapping via ARASAAC.

Real AAC practice uses established, clinically-vetted symbol sets rather
than freshly generated illustrations — consistency of symbols across a
child's boards matters more than novelty. ARASAAC (https://arasaac.org) is
a free, openly-licensed AAC pictogram set with a public search API, so we
map each generated word to a real symbol instead of inventing new art.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import requests
from pydantic import BaseModel

_MAX_WORKERS = 10

ARASAAC_SEARCH_URL = "https://api.arasaac.org/api/pictograms/en/search/{query}"
ARASAAC_IMAGE_URL = "https://static.arasaac.org/pictograms/{symbol_id}/{symbol_id}_500.png"
_TIMEOUT = 5


class SymbolEntry(BaseModel):
    word: str
    symbol_id: int | None = None
    image_url: str | None = None


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


def _search(term: str) -> int | None:
    try:
        resp = requests.get(ARASAAC_SEARCH_URL.format(query=term), timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        return None

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


@lru_cache(maxsize=512)
def find_symbol(word: str) -> SymbolEntry:
    """Look up a real AAC symbol for `word`. Falls back to the first word of
    a multi-word phrase if the exact phrase has no match, and returns a
    symbol-less entry (word only) if nothing is found at all — the board
    still shows the word as text rather than silently dropping it."""
    symbol_id = _search(word)
    if symbol_id is None and " " in word:
        symbol_id = _search(word.split(" ")[0])

    if symbol_id is None:
        return SymbolEntry(word=word)

    return SymbolEntry(word=word, symbol_id=symbol_id, image_url=ARASAAC_IMAGE_URL.format(symbol_id=symbol_id))


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
