"""Persists full board records (photo + caption + generated vocabulary)
per child, so a "My Library" view can list and reopen past boards without
re-running the whole pipeline — this is the missing piece Phase 5 called
out (history.py only ever tracked vocabulary words, not full boards).
"""

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scenespeak.db"
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS boards (
            id TEXT PRIMARY KEY,
            child_id TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            caption TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    return conn


def save_board(child_id: str, image_bytes: bytes, image_ext: str, caption: str, board_data: dict) -> dict:
    board_id = uuid.uuid4().hex
    image_filename = f"{board_id}{image_ext}"
    (UPLOADS_DIR / image_filename).write_bytes(image_bytes)

    created_at = datetime.now(timezone.utc).isoformat()
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO boards (id, child_id, image_filename, caption, data_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (board_id, child_id, image_filename, caption, json.dumps(board_data), created_at),
        )
        conn.commit()

    return {"id": board_id, "image_url": f"/uploads/{image_filename}", "caption": caption, "created_at": created_at}


def list_boards(child_id: str) -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, image_filename, caption, created_at FROM boards WHERE child_id = ? ORDER BY created_at DESC",
            (child_id,),
        ).fetchall()
    return [
        {"id": r[0], "image_url": f"/uploads/{r[1]}", "caption": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_board(board_id: str, owner: str) -> dict | None:
    """Return a board only if it belongs to `owner` (see app/owner.py) —
    the ownership check is part of the query, so another owner's board is
    indistinguishable from a missing one."""
    prefix = f"{owner}:"
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, image_filename, caption, data_json, created_at, child_id FROM boards "
            "WHERE id = ? AND substr(child_id, 1, ?) = ?",
            (board_id, len(prefix), prefix),
        ).fetchone()
    if row is None:
        return None
    data = json.loads(row[3])
    data["id"] = row[0]
    data["image_url"] = f"/uploads/{row[1]}"
    data["caption"] = row[2]
    data["created_at"] = row[4]
    data["child_id"] = row[5]
    return data


# Fields stored in data_json; the rest of a board (id, image, caption, date)
# lives in its own columns.
_META_FIELDS = {"id", "image_url", "caption", "created_at", "child_id", "timings_ms"}


def update_board(board_id: str, owner: str, board: dict) -> None:
    """Save an edited board's words (owner-checked like get_board)."""
    prefix = f"{owner}:"
    data = {k: v for k, v in board.items() if k not in _META_FIELDS}
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE boards SET data_json = ? WHERE id = ? AND substr(child_id, 1, ?) = ?",
            (json.dumps(data), board_id, len(prefix), prefix),
        )
        conn.commit()


def board_words(board_ids: list[str], categories: tuple[str, ...]) -> dict[str, list[str]]:
    """The words currently on each of these boards, by board id."""
    if not board_ids:
        return {}
    marks = ",".join("?" * len(board_ids))
    with closing(_connect()) as conn:
        rows = conn.execute(f"SELECT id, data_json FROM boards WHERE id IN ({marks})", board_ids).fetchall()
    out = {}
    for board_id, data_json in rows:
        data = json.loads(data_json)
        out[board_id] = [e["word"] for c in categories for e in data.get(c, [])]
    return out
