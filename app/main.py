"""SceneSpeak — photo in, a symbol-mapped, RAG-grounded AAC board out.
Run with: python run.py
"""

import io
import logging
import mimetypes
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app.boards import UPLOADS_DIR, get_board, list_boards, save_board
from app.captioning import generate_caption
from app.history import add_to_history, get_child_history
from app.rag import retrieve_vocabulary_context
from app.symbols import SymbolEntry, map_words_to_symbols
from app import usage
from app.tts import TTSError, synthesize
from app.verification import get_verifier, verify_board
from app.vocab import FITZGERALD_ORDER, GeminiBusyError, GeminiKeyError, GeminiQuotaError, generate_vocabulary

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s %(message)s")

app = FastAPI(title="SceneSpeak", version="0.7.0")


@app.on_event("startup")
def _warm_up():
    """Load BLIP and build/open the RAG index in the background at startup,
    so the first caregiver's board doesn't pay for it (measured: ~8s BLIP
    load, ~30s one-time index build on a fresh install)."""
    def warm():
        blank = Image.new("RGB", (64, 64))
        generate_caption(blank)
        retrieve_vocabulary_context("warm up", None, k=1)
        get_verifier().verify(blank, ["ball"], caption="a ball")  # loads its model/data, if any
    threading.Thread(target=warm, daemon=True).start()


class NoCacheStaticFiles(StaticFiles):
    """Plain StaticFiles lets browsers cache CSS/JS aggressively with no
    revalidation, which meant a redesign could silently keep showing an old
    stylesheet after a refresh (this happened during development — the
    layout-affecting rules were from a stale cached file while newer,
    unrelated rules loaded fine, which is a very confusing bug to diagnose
    from the outside). This forces a revalidation check on every request
    instead of trusting a stale local copy."""

    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


@app.get("/")
def index():
    # Same reason as NoCacheStaticFiles: without this a browser can keep
    # showing an old homepage after the HTML changes.
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


class BoardData(BaseModel):
    caption: str
    # Fitzgerald Key order (see app/vocab.py). Defaults keep boards saved
    # before this layout loadable.
    pronouns: list[SymbolEntry] = []
    verbs: list[SymbolEntry] = []
    prepositions: list[SymbolEntry] = []
    descriptors: list[SymbolEntry] = []
    objects: list[SymbolEntry] = []
    core: list[SymbolEntry] = []  # legacy: boards generated before the QuickPic-style switch
    rejected_objects: list[dict] = []  # legacy: CLIP-filtered words on boards from before CLIP was removed
    verification: dict | None = None  # which verifier ran, its latency, how many words it scored


class BoardResponse(BoardData):
    id: str
    image_url: str
    created_at: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _server_has_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


@app.get("/config")
def config():
    """Lets the frontend hide the API-key field when the server already
    holds a key, and show how many boards are left today."""
    has_key = _server_has_key()
    return {
        "server_key": has_key,
        "daily_limit": usage.daily_limit() if has_key else None,
        "remaining_today": usage.remaining_today() if has_key else None,
    }


@app.get("/tts")
def tts(text: str):
    """Spoken audio (MP3) for a board word. Cached per word, so the browser
    can also cache it long-term — the same text always yields the same audio."""
    try:
        path = synthesize(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TTSError as exc:
        # The frontend falls back to the browser's Web Speech API on any
        # non-OK response, so a gTTS outage degrades instead of going silent.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return FileResponse(path, media_type="audio/mpeg",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/boards")
def get_boards(child_id: str = "default"):
    return list_boards(child_id)


@app.get("/boards/{board_id}", response_model=BoardResponse)
def get_one_board(board_id: str):
    board = get_board(board_id)
    if board is None:
        raise HTTPException(status_code=404, detail="Board not found.")
    return board


@app.post("/generate-board", response_model=BoardResponse)
async def generate_board(
    photo: UploadFile = File(..., description="A real photo of the scene/activity."),
    scenario: str | None = Form(None, description="Optional context, e.g. 'practicing at the playground'."),
    child_id: str = Form("default", description="Identifies the child, so vocabulary stays consistent across their boards."),
    gemini_api_key: str | None = Form(None, description="Optional — overrides GEMINI_API_KEY from the environment for this request."),
):
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await photo.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    # Only boards that spend the server's own key count toward the daily
    # cap. Checked before BLIP/RAG so a capped request fails fast.
    uses_server_key = not gemini_api_key and _server_has_key()
    if uses_server_key and not usage.try_reserve():
        raise HTTPException(
            status_code=429,
            detail=f"Today's limit of {usage.daily_limit()} boards has been reached. "
                   "It resets at midnight UTC — please try again tomorrow.",
        )

    try:
        caption = generate_caption(image)
        retrieved_vocabulary = retrieve_vocabulary_context(caption, scenario)
        child_history = get_child_history(child_id)
        vocabulary = generate_vocabulary(
            caption,
            scenario,
            image=image,
            retrieved_vocabulary=retrieved_vocabulary,
            child_history=child_history,
            api_key=gemini_api_key or None,
        )
    except Exception as exc:
        # No usable board came out of this request, so don't let it eat
        # into today's allowance.
        if uses_server_key:
            usage.release()
        if isinstance(exc, GeminiQuotaError):
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        if isinstance(exc, GeminiBusyError):
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if isinstance(exc, GeminiKeyError):
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        if isinstance(exc, RuntimeError):
            # e.g. missing GEMINI_API_KEY — surface the real reason as JSON so
            # the frontend can show it, instead of a generic 500 with no body.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=f"Vocabulary generation failed: {exc}") from exc

    vocab_dict = vocabulary.model_dump()
    add_to_history(child_id, vocab_dict)

    # Symbol lookups are the actual bottleneck in board generation (each
    # word is a separate network round-trip to ARASAAC, measured at 1s+ per
    # word) -- combine every category's words into a single parallel batch
    # instead of separate sequential batches per category, so all of them are
    # in flight together rather than one category waiting on the last.
    categories = {c: vocab_dict[c] for c in FITZGERALD_ORDER}
    all_words = list(dict.fromkeys(w for words in categories.values() for w in words))

    # Verification (CPU) and symbol lookups (network) don't depend on each
    # other, so run them side by side rather than one after the other.
    verifier = get_verifier()
    with ThreadPoolExecutor(max_workers=1) as pool:
        verification = pool.submit(verify_board, image, categories, caption, verifier)
        symbols_by_word = dict(zip(all_words, map_words_to_symbols(all_words)))
        scores, verify_ms = verification.result()

    def entry(word: str) -> SymbolEntry:
        # Copy: SymbolEntry objects are shared through find_symbol's cache.
        s = scores[word]
        return symbols_by_word[word].model_copy(update={"score": s.score, "low_confidence": not s.verified})

    board_data = BoardData(
        caption=caption,
        **{c: [entry(w) for w in words] for c, words in categories.items()},
        verification={"verifier": verifier.name, "latency_ms": round(verify_ms),
                      "checked": sum(1 for s in scores.values() if s.score is not None)},
    )

    ext = mimetypes.guess_extension(photo.content_type) or ".jpg"
    saved = save_board(child_id, raw, ext, caption, board_data.model_dump())

    return BoardResponse(**board_data.model_dump(), id=saved["id"], image_url=saved["image_url"], created_at=saved["created_at"])
