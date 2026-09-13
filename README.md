# SceneSpeak — AI-Generated Communication Boards for Autistic and Nonverbal Children

Generates AAC (Augmentative and Alternative Communication) topic boards
from a real photo of a scene, for speech-language pathologists and
caregivers supporting nonverbal or minimally-verbal children.

Extends the approach explored in academic work on LLM-generated AAC
vocabulary (e.g. QuickPic, CHI 2024) with an added reliability layer:
generated vocabulary is cross-checked against the actual photo before it
reaches the user.

## Status: Phase 4 done, Phase 6 (frontend) started — see `PLAN.md`

Open http://127.0.0.1:8000/ (not just `/docs`) for the actual UI: upload a
photo, get back a board grouped by part of speech, tap any symbol to hear
it spoken (via the browser's built-in speech synthesis — no backend TTS
service needed for this), with a transparent note about anything the
verification layer filtered out.


Working right now: upload a photo → BLIP captions it → relevant vocabulary
is retrieved from a curated AAC corpus (ChromaDB) and combined with that
child's prior history (SQLite) → an LLM generates categorized AAC
vocabulary grounded in all of that → generated **objects** are
independently verified against the real photo with CLIP, filtering out
anything that doesn't actually match (measured **90.9% precision / 83.3%
recall** — see `scripts/evaluate_verification.py`) → surviving words are
mapped to real ARASAAC pictograms (falling back to text-only when no good
symbol exists, rather than showing a wrong one).

Not yet built: TTS + persistence polish (Phase 5), frontend + deployment
(Phase 6).

## Stack (Phase 1–4, 6)
- FastAPI backend
- BLIP (`Salesforce/blip-image-captioning-base`) for scene captioning
- ChromaDB for RAG retrieval over the AAC vocabulary corpus (`data/`)
- SQLite for per-child vocabulary history
- Gemini API for vocabulary generation
- CLIP (`openai/clip-vit-base-patch32`) for vision-language verification
- ARASAAC public API for AAC symbol mapping

## Evaluating the verification layer
```bash
python scripts/build_test_images.py       # builds the labeled synthetic test set
python scripts/evaluate_verification.py   # reports precision/recall/accuracy across thresholds
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

You need a [Gemini API key](https://aistudio.google.com/apikey). Either:
- copy `.env.example` to `.env` and set `GEMINI_API_KEY` there, **or**
- paste it into the "Gemini API key" field on the web UI itself — it's saved in your browser's local storage (not committed, not sent anywhere but this app's own backend) so you only enter it once per browser.

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
