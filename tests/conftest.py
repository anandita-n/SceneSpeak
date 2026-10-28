"""Shared fixtures. The board pipeline's models and network calls are
stubbed out, so the API tests run in seconds, offline, with no Gemini quota
used — they test SceneSpeak's own logic, not BLIP or Gemini."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OWNER_A = "A" * 32
OWNER_B = "B" * 32


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Point every SQLite table and the uploads folder at a temp directory."""
    from app import boards, history, symbols, usage

    db = tmp_path / "test.db"
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    for module in (boards, history, symbols, usage):
        monkeypatch.setattr(module, "DB_PATH", db)
    monkeypatch.setattr(boards, "UPLOADS_DIR", uploads)
    return tmp_path


@pytest.fixture
def client(isolated_storage, monkeypatch):
    """A TestClient whose board pipeline returns a fixed board."""
    from fastapi.testclient import TestClient

    import app.main as main
    from app.symbols import SymbolEntry
    from app.verification import NoVerifier
    from app.vocab import VocabularySet

    monkeypatch.setattr(main, "generate_caption", lambda image: "a kitchen with a fridge")
    monkeypatch.setattr(main, "retrieve_vocabulary_context", lambda caption, scenario: [])
    monkeypatch.setattr(main, "generate_vocabulary", lambda *a, **k: VocabularySet(
        pronouns=["I"], verbs=["cook"], prepositions=["in"], descriptors=["hot"], objects=["fridge", "oven"]))
    monkeypatch.setattr(main, "map_words_to_symbols", lambda words: [SymbolEntry(word=w) for w in words])
    monkeypatch.setattr(main, "get_verifier", lambda name=None: NoVerifier())
    monkeypatch.setattr(main, "find_symbol", lambda w: SymbolEntry(word=w))

    # Each test gets its own in-memory Chroma collection for child scenes.
    import uuid

    import chromadb
    from chromadb.config import Settings

    from app import personalization

    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    scenes = client.get_or_create_collection(f"scenes_{uuid.uuid4().hex}")
    monkeypatch.setattr(personalization, "_scenes", lambda: scenes)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # no daily-limit bookkeeping in these tests
    return TestClient(main.app)


@pytest.fixture
def photo_bytes():
    """A tiny valid PNG."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()
