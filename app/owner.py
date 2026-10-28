"""Private per-browser ownership of boards.

There are no accounts. Each browser creates a random secret key once
(static/app.js) and sends it as the X-Owner-Key header. Every child's
boards, photos list and word history are stored under a hash of that key,
so another visitor who types the same child's name sees nothing — and a
leaked database doesn't reveal the keys themselves.

Losing the key (clearing browser data, or switching device) loses access to
those boards; that is the trade-off for not needing a sign-up.
"""

import hashlib
import re

from fastapi import Header, HTTPException

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{22,128}$")  # >= 128 bits of entropy from the client


def owner_scope(x_owner_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: validate the X-Owner-Key header and return the
    owner's storage scope (a hash of the key)."""
    if not x_owner_key or not _KEY_PATTERN.match(x_owner_key):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid owner key. Reload the page to create one.",
        )
    return "o" + hashlib.sha256(x_owner_key.encode("utf-8")).hexdigest()[:32]


def scoped_child(owner: str, child_id: str) -> str:
    """The storage id for one child under one owner."""
    return f"{owner}:{child_id.strip() or 'default'}"
