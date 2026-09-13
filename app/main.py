"""SceneSpeak — Phase 1: photo in, BLIP caption + LLM-generated AAC
vocabulary out. Run with: uvicorn app.main:app --reload
"""

import io

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from PIL import Image

from app.captioning import generate_caption
from app.vocab import VocabularySet, generate_vocabulary

load_dotenv()

app = FastAPI(title="SceneSpeak", version="0.1.0")


class BoardResponse(VocabularySet):
    caption: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-board", response_model=BoardResponse)
async def generate_board(
    photo: UploadFile = File(..., description="A real photo of the scene/activity."),
    scenario: str | None = Form(None, description="Optional context, e.g. 'practicing at the playground'."),
):
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await photo.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    caption = generate_caption(image)
    vocabulary = generate_vocabulary(caption, scenario)

    return BoardResponse(caption=caption, **vocabulary.model_dump())
