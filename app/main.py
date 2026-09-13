"""SceneSpeak — photo in, a symbol-mapped, RAG-grounded AAC board out.
Run with: uvicorn app.main:app --reload
"""

import io

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from PIL import Image
from pydantic import BaseModel

from app.captioning import generate_caption
from app.history import add_to_history, get_child_history
from app.rag import retrieve_vocabulary_context
from app.symbols import SymbolEntry, map_words_to_symbols
from app.vocab import generate_vocabulary

load_dotenv()

app = FastAPI(title="SceneSpeak", version="0.3.0")


class BoardResponse(BaseModel):
    caption: str
    core: list[SymbolEntry]
    objects: list[SymbolEntry]
    descriptors: list[SymbolEntry]
    prepositions: list[SymbolEntry]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-board", response_model=BoardResponse)
async def generate_board(
    photo: UploadFile = File(..., description="A real photo of the scene/activity."),
    scenario: str | None = Form(None, description="Optional context, e.g. 'practicing at the playground'."),
    child_id: str = Form("default", description="Identifies the child, so vocabulary stays consistent across their boards."),
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

    vocabulary = generate_vocabulary(
        caption,
        scenario,
        retrieved_vocabulary=retrieved_vocabulary,
        child_history=child_history,
    )
    vocab_dict = vocabulary.model_dump()

    add_to_history(child_id, vocab_dict)

    return BoardResponse(
        caption=caption,
        core=map_words_to_symbols(vocab_dict["core"]),
        objects=map_words_to_symbols(vocab_dict["objects"]),
        descriptors=map_words_to_symbols(vocab_dict["descriptors"]),
        prepositions=map_words_to_symbols(vocab_dict["prepositions"]),
    )
