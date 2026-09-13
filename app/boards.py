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


def get_board(board_id: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, image_filename, caption, data_json, created_at FROM boards WHERE id = ?",
            (board_id,),
        ).fetchone()
    if row is None:
        return None
    data = json.loads(row[3])
    data["id"] = row[0]
    data["image_url"] = f"/uploads/{row[1]}"
    data["caption"] = row[2]
    data["created_at"] = row[4]
    return data
