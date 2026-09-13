"""Phase 2: RAG grounding over a curated AAC vocabulary corpus.

Instead of letting the LLM freely invent vocabulary from the caption alone,
we retrieve the most semantically relevant entries from a real,
standardized AAC core-vocabulary list plus scenario-tagged fringe
vocabulary, and feed those into the generation prompt. This keeps
suggestions consistent with actual AAC practice instead of varying
arbitrarily between runs.
"""

import json
from functools import lru_cache
from pathlib import Path

import chromadb

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "aac_vocabulary"


def _load_corpus() -> list[dict]:
    entries = []
    with open(DATA_DIR / "aac_core_vocabulary.json", encoding="utf-8") as f:
        for item in json.load(f):
            entries.append({**item, "scenario": "core"})
    with open(DATA_DIR / "aac_scenario_vocabulary.json", encoding="utf-8") as f:
        entries.extend(json.load(f))
    return entries


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    if collection.count() == 0:
        entries = _load_corpus()
        # Embed each entry as "<word> (<scenario> <part_of_speech>)" so the
        # embedding captures both the word and its usage context.
        documents = [f"{e['word']} ({e['scenario']} {e['part_of_speech']})" for e in entries]
        ids = [f"vocab-{i}" for i in range(len(entries))]
        metadatas = entries
        collection.add(documents=documents, ids=ids, metadatas=metadatas)

    return collection


def retrieve_vocabulary_context(caption: str, scenario: str | None, k: int = 15) -> list[dict]:
    """Return up to `k` corpus entries most relevant to this caption/scenario,
    always including the full core-vocabulary set (it's small and should
    always be available as an option, not just when semantically close)."""
    collection = _get_collection()
    query = caption if not scenario else f"{caption}. Context: {scenario}"
    results = collection.query(query_texts=[query], n_results=k)
    retrieved = results["metadatas"][0] if results["metadatas"] else []

    # Always include core vocabulary, de-duplicated against retrieved results.
    core = [e for e in _load_corpus() if e["scenario"] == "core"]
    seen_words = {e["word"] for e in retrieved}
    for entry in core:
        if entry["word"] not in seen_words:
            retrieved.append(entry)
            seen_words.add(entry["word"])

    return retrieved
