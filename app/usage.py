"""Daily usage cap for boards generated with the server's own Gemini key.

Once deployed, the key lives on the server (never in the UI), so anyone
who finds the URL is spending the project's quota. This caps how many
boards can use that key per day (UTC), counted in SQLite so a server
restart doesn't reset it. Requests that bring their own key aren't counted —
that's the caller's quota, not ours.
"""

import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scenespeak.db"
DEFAULT_DAILY_LIMIT = 30


def daily_limit() -> int:
    try:
        return max(0, int(os.environ.get("DAILY_BOARD_LIMIT", DEFAULT_DAILY_LIMIT)))
    except ValueError:
        return DEFAULT_DAILY_LIMIT


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS daily_usage (
            day TEXT PRIMARY KEY,
            count INTEGER NOT NULL
        )"""
    )
    return conn


def used_today() -> int:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT count FROM daily_usage WHERE day = ?", (_today(),)).fetchone()
    return row[0] if row else 0


def remaining_today() -> int:
    return max(0, daily_limit() - used_today())


def try_reserve() -> bool:
    """Atomically take one slot from today's allowance. Returns False if
    the limit is already reached. The conditional UPDATE means two
    concurrent requests can't both take the last slot."""
    day, limit = _today(), daily_limit()
    with closing(_connect()) as conn:
        conn.execute("INSERT OR IGNORE INTO daily_usage (day, count) VALUES (?, 0)", (day,))
        cur = conn.execute(
            "UPDATE daily_usage SET count = count + 1 WHERE day = ? AND count < ?", (day, limit)
        )
        conn.commit()
        return cur.rowcount == 1


def release() -> None:
    """Give a reserved slot back — used when generation fails before the
    key was actually charged for a usable board."""
    with closing(_connect()) as conn:
        conn.execute(
            "UPDATE daily_usage SET count = count - 1 WHERE day = ? AND count > 0", (_today(),)
        )
        conn.commit()
