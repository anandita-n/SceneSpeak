"""Phase 5: server-side text-to-speech via gTTS.

The browser's Web Speech API sounds different on every device/OS (and is
missing entirely on some), which is a real problem for AAC: a child relies
on a word sounding the same every time they tap it. Generating audio on
the backend gives one consistent voice everywhere. Board vocabulary is a
small, highly repetitive set of words, so each word is synthesized once and
cached to disk as an MP3 — repeat taps (and every other board that uses
the same word) are served straight from the cache with no network call.
"""

import hashlib
import re
from pathlib import Path

from gtts import gTTS
from gtts.tts import gTTSError

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "tts_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_TEXT_LENGTH = 100  # board tiles are single words/short phrases, not paragraphs
_LANG = "en"
_SLOW = False


class TTSError(RuntimeError):
    pass


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _cache_path(text: str) -> Path:
    key = hashlib.sha256(f"{_LANG}|{_SLOW}|{text}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.mp3"


def synthesize(text: str) -> Path:
    """Return the path to an MP3 of `text` spoken aloud, generating and
    caching it on first request."""
    text = _normalize(text)
    if not text:
        raise ValueError("Text is empty.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text is longer than {MAX_TEXT_LENGTH} characters.")

    path = _cache_path(text)
    if path.exists():
        return path

    # Write to a temp file then rename, so a concurrent request (or a failed
    # synthesis) never leaves a half-written MP3 in the cache.
    tmp = path.with_suffix(".tmp")
    try:
        gTTS(text=text, lang=_LANG, slow=_SLOW).save(str(tmp))
        tmp.replace(path)
    except gTTSError as exc:
        tmp.unlink(missing_ok=True)
        raise TTSError(f"Speech synthesis failed: {exc}") from exc
    return path
