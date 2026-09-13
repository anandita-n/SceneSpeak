# SceneSpeak — AI-Generated Communication Boards for Autistic and Nonverbal Children

Generates AAC (Augmentative and Alternative Communication) topic boards
from a real photo of a scene, for speech-language pathologists and
caregivers supporting nonverbal or minimally-verbal children.

Extends the approach explored in academic work on LLM-generated AAC
vocabulary (e.g. QuickPic, CHI 2024) with an added reliability layer:
generated vocabulary is cross-checked against the actual photo before it
reaches the user.

## Status: Phase 2 (of 6) — see `PLAN.md`

Working right now: upload a photo → BLIP captions it → the most relevant
entries are retrieved from a curated AAC vocabulary corpus (ChromaDB) and
combined with that child's prior vocabulary history (SQLite) → an LLM
generates categorized AAC vocabulary (core words, objects, descriptors,
prepositions) grounded in all of that, not just the raw caption.

Not yet built: symbol mapping to an existing AAC symbol set (Phase 3),
BLIP/CLIP-based verification of generated vocabulary (Phase 4), TTS +
persistence polish (Phase 5), frontend + deployment (Phase 6).

## Stack (Phase 1–2)
- FastAPI backend
- BLIP (`Salesforce/blip-image-captioning-base`) for scene captioning
- ChromaDB for RAG retrieval over the AAC vocabulary corpus (`data/`)
- SQLite for per-child vocabulary history
- Gemini API for vocabulary generation

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
copy .env.example .env                              # then add your GEMINI_API_KEY
```

## Run
```bash
uvicorn app.main:app --reload
```

## Try it
```bash
curl -X POST http://127.0.0.1:8000/generate-board ^
  -F "photo=@sample.jpg" ^
  -F "scenario=practicing at the playground"
```

Or open http://127.0.0.1:8000/docs for the interactive Swagger UI.
