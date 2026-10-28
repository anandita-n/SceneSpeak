"""Personalization from caregiver edits: retrieval over each child's own boards.

Every saved board's scene (caption + optional context) is indexed in Chroma
under the child's id. When a new board is made for that child, the most
similar past scenes are retrieved, and from those boards:

- words the caregiver REMOVED are excluded from the new board,
- words the caregiver ADDED are passed to the model to include if they fit,
- the remaining words set the child's established wording.

This replaces pasting the child's entire (ever-growing) word history into
every prompt. Retrieval over a user's own content is how LLM personalization
is usually done (e.g. the AAC system in "Empowering AAC Users", CustomNLP4U
2024); the generic AAC word sets in app/rag.py now only set standard wording.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app import rag

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scenespeak.db"
SCENES_COLLECTION = "child_scenes"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS board_edits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id TEXT NOT NULL,
            board_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK (action IN ('add', 'remove')),
            word TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    return conn


def _scenes():
    return rag.get_client().get_or_create_collection(SCENES_COLLECTION)


def scene_text(caption: str, scenario: str | None) -> str:
    return f"{caption}. Context: {scenario}" if scenario else caption


def index_board(child_id: str, board_id: str, caption: str, scenario: str | None) -> None:
    """Make a saved board findable when the same child gets a similar scene."""
    _scenes().upsert(ids=[board_id], documents=[scene_text(caption, scenario)],
                     metadatas=[{"child_id": child_id}])


def similar_boards(child_id: str, caption: str, scenario: str | None,
                   n: int = 3, max_distance: float = 1.0) -> list[str]:
    """Ids of this child's boards for the most similar past scenes."""
    collection = _scenes()
    try:
        result = collection.query(query_texts=[scene_text(caption, scenario)], n_results=n,
                                  where={"child_id": child_id})
    except Exception:  # no boards indexed yet for anyone / this child
        return []
    ids = result["ids"][0] if result["ids"] else []
    distances = result["distances"][0] if result.get("distances") else [0.0] * len(ids)
    return [i for i, d in zip(ids, distances) if d <= max_distance]


def record_edit(child_id: str, board_id: str, action: str, word: str, category: str) -> None:
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO board_edits (child_id, board_id, action, word, category, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (child_id, board_id, action, word, category, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def edits_for(board_ids: list[str]) -> tuple[list[str], list[str]]:
    """(added, removed) words across these boards. A word's latest edit wins,
    so removing and then re-adding a word counts as added."""
    if not board_ids:
        return [], []
    marks = ",".join("?" * len(board_ids))
    with closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT action, word FROM board_edits WHERE board_id IN ({marks}) ORDER BY id", board_ids
        ).fetchall()
    latest: dict[str, tuple[str, str]] = {}
    for action, word in rows:
        latest[word.lower()] = (action, word)
    added = [w for a, w in latest.values() if a == "add"]
    removed = [w for a, w in latest.values() if a == "remove"]
    return added, removed


def drop_removed(vocab: dict[str, list[str]], removed: list[str]) -> dict[str, list[str]]:
    """Remove caregiver-removed words from a generated board (case-insensitive)."""
    gone = {w.lower() for w in removed}
    return {c: [w for w in words if w.lower() not in gone] for c, words in vocab.items()}
