"""Per-child vocabulary history — keeps a child's previously-used words
consistent across sessions, matching how real AAC boards rely on stable
word placement/wording for motor planning (a word shouldn't be phrased
differently every time the same scenario comes up).
"""

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scenespeak.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS child_vocabulary (
            child_id TEXT NOT NULL,
            word TEXT NOT NULL,
            category TEXT NOT NULL,
            PRIMARY KEY (child_id, word)
        )"""
    )
    return conn


def get_child_history(child_id: str) -> list[str]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT word FROM child_vocabulary WHERE child_id = ?", (child_id,)
        ).fetchall()
    return [r[0] for r in rows]


def add_to_history(child_id: str, words_by_category: dict[str, list[str]]) -> None:
    with closing(_connect()) as conn:
        for category, words in words_by_category.items():
            conn.executemany(
                "INSERT OR IGNORE INTO child_vocabulary (child_id, word, category) VALUES (?, ?, ?)",
                [(child_id, w, category) for w in words],
            )
        conn.commit()
