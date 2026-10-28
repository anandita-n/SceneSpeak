"""Phase 2: RAG grounding over curated AAC scenario word sets.

Prior LLM-based AAC work (QuickPic AAC, Yu et al. 2024) generates each topic display from
scratch, so the same scene can yield different wording every time. Instead
we retrieve the most semantically relevant entries from curated,
scenario-tagged AAC word sets (playground, mealtime, classroom...) and feed
them into the generation prompt — together with the child's own board
history (app/history.py) — so suggestions stay consistent across runs and
across a child's boards.

The corpus covers 36 everyday scenarios (~800 entries, ~500 unique words),
far more than belongs in one prompt, so retrieval narrows it to the
handful of scenarios a given photo is actually about.
"""

import json
from functools import lru_cache
from pathlib import Path

import logging

import chromadb
from chromadb.config import Settings

# chromadb 0.5.5 still calls its (incompatible) posthog client even with
# telemetry off, logging a harmless "Failed to send telemetry event" error
# on every operation. Silence just that logger.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
# Versioned name: the corpus changed (core list removed, phrases relabeled),
# so a new name makes existing installs rebuild the index instead of
# silently querying stale embeddings.
COLLECTION_NAME = "aac_scenario_vocabulary_v3"


def _load_corpus() -> list[dict]:
    with open(DATA_DIR / "aac_scenario_vocabulary.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_client():
    """The one Chroma client for this process (shared with app/personalization.py —
    Chroma doesn't allow two clients on the same folder)."""
    # Telemetry off: it's not needed, and chromadb 0.5.5's telemetry client
    # only logs "Failed to send telemetry event" errors with current posthog.
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


@lru_cache(maxsize=1)
def _get_collection():
    collection = get_client().get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        entries = _load_corpus()
        # Embed each entry as "<word> (<scenario> <part_of_speech>)" so the
        # embedding captures both the word and its usage context.
        documents = [f"{e['word']} ({e['scenario']} {e['part_of_speech']})" for e in entries]
        ids = [f"vocab-{i}" for i in range(len(entries))]
        metadatas = entries
        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    return collection


def retrieve_vocabulary_context(caption: str, scenario: str | None, k: int = 30) -> list[dict]:
    """Return up to `k` scenario-vocabulary entries most relevant to this
    caption/scenario."""
    collection = _get_collection()
    query = caption if not scenario else f"{caption}. Context: {scenario}"
    results = collection.query(query_texts=[query], n_results=k)
    return results["metadatas"][0] if results["metadatas"] else []
