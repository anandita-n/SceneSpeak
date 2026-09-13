"""SceneSpeak — photo in, a verified, symbol-mapped, RAG-grounded AAC
board out. Run with: uvicorn app.main:app --reload
"""

import io

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from app.captioning import generate_caption
from app.history import add_to_history, get_child_history
from app.rag import retrieve_vocabulary_context
from app.symbols import SymbolEntry, map_words_to_symbols
from app.verification import verify_objects
from app.vocab import generate_vocabulary

load_dotenv()

app = FastAPI(title="SceneSpeak", version="0.5.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


class BoardResponse(BaseModel):
    caption: str
    core: list[SymbolEntry]
    objects: list[SymbolEntry]
    descriptors: list[SymbolEntry]
    prepositions: list[SymbolEntry]
    rejected_objects: list[dict]  # words the LLM suggested that verification filtered out, with their CLIP scores


@app.get("/health")
def health():
    return {"status": "ok"}


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

    caption = generate_caption(image)
    retrieved_vocabulary = retrieve_vocabulary_context(caption, scenario)
    child_history = get_child_history(child_id)

    try:
        vocabulary = generate_vocabulary(
            caption,
            scenario,
            retrieved_vocabulary=retrieved_vocabulary,
            child_history=child_history,
            api_key=gemini_api_key or None,
        )
    except RuntimeError as exc:
        # e.g. missing GEMINI_API_KEY — surface the real reason as JSON so
        # the frontend can show it, instead of a generic 500 with no body.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Vocabulary generation failed: {exc}") from exc

    vocab_dict = vocabulary.model_dump()

    # Verify only the "objects" category — core vocabulary and prepositions
    # aren't things a photo depicts, so scoring them against image
    # similarity wouldn't mean anything (see app/verification.py).
    object_scores = verify_objects(image, vocab_dict["objects"])
    verified_objects = [r["word"] for r in object_scores if r["passed"]]
    rejected_objects = [r for r in object_scores if not r["passed"]]
    vocab_dict["objects"] = verified_objects

    add_to_history(child_id, vocab_dict)

    return BoardResponse(
        caption=caption,
        core=map_words_to_symbols(vocab_dict["core"]),
        objects=map_words_to_symbols(verified_objects),
        descriptors=map_words_to_symbols(vocab_dict["descriptors"]),
        prepositions=map_words_to_symbols(vocab_dict["prepositions"]),
        rejected_objects=rejected_objects,
    )
