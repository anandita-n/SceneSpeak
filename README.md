# SceneSpeak — AI-Generated Communication Boards for Autistic and Nonverbal Children

Generates AAC (Augmentative and Alternative Communication) topic boards
from a real photo of a scene, for speech-language pathologists and
caregivers supporting nonverbal or minimally-verbal children.

Builds on QuickPic AAC (Yu et al., *IJERPH* 2024), which generates
topic-specific vocabulary per part of speech with an LLM and lays it out in
modified Fitzgerald Key order, and on Zastudil et al. (2024), who gave a
multimodal model the photo itself. SceneSpeak adds **RAG grounding** that
QuickPic doesn't have: suggestions are grounded in curated AAC word sets for
36 everyday scenarios and in the child's own board history, so wording stays
consistent across boards instead of being regenerated from scratch each time.

## Status: all phases done except deployment — see `PLAN.md`

Open http://127.0.0.1:8000/ for the actual app:
- **Home** — Create a Board / My Library
- **Create a Board** — upload a photo, optional context, child name
- **Board** — the photo and the generated symbol board side by side, tap any symbol to hear it spoken, Edit mode to remove a wrongly-suggested tile
- **My Library** — every board you've created for that child, as a photo grid; click one to reopen it

## Pipeline
1. **BLIP** (vision-language model) captions the photo.
2. **RAG**: the caption is the query into ChromaDB over ~800 curated AAC
   entries (36 scenarios, ~500 unique words) — retrieval narrows that to the
   few scenarios the photo is about — plus the child's prior words (SQLite).
3. **Gemini** (multimodal) receives the **photo itself**, the caption, the
   retrieved words and the child's history, and returns pronouns, verbs,
   prepositions, descriptors and objects as JSON. Seeing the photo means it
   doesn't have to guess which objects a scene "probably" contains — the
   co-occurrence hallucination POPE (Li et al., 2023) documents.
4. Words are mapped to **ARASAAC** pictograms (text-only fallback rather than
   a wrong symbol).
5. The board is laid out in modified **Fitzgerald Key** columns (people,
   actions, prepositions, describing words, things — left to right,
   colour-coded by the usual convention).
6. Tapping a tile speaks it via **gTTS** (`GET /tts`, cached per word), with
   the browser's Web Speech API as fallback.
7. The caregiver can remove any wrong tile in **Edit mode** — the same
   human-review safety net QuickPic relies on.

There's deliberately no generated "core words" category: a child's core
vocabulary lives in fixed positions on their main AAC device (motor
planning depends on that), so a topic board adds scene-specific words
rather than regenerating core ones.

An earlier version filtered objects with CLIP image–text similarity. On
real photos it rejected clearly visible objects (desk, chair, book, pencil
in a classroom) while scoring an absent "spoon" above a present "hand" —
its single threshold had been tuned on synthetic drawings. It was removed
in favour of giving Gemini the photo directly.

Typical board generation: ~8–12 s (mostly Gemini + ARASAAC lookups). BLIP
and the RAG index are warmed up at server start.

Not yet built: Docker + deployment (the server-side key + daily limit it needs are done).

## Stack
- FastAPI backend
- BLIP (`Salesforce/blip-image-captioning-base`) for scene captioning
- ChromaDB for RAG retrieval over curated AAC scenario word sets (`data/aac_scenario_vocabulary.json`)
- SQLite for per-child vocabulary history, saved boards and the daily usage counter
- Gemini (multimodal, `gemini-flash-lite-latest` with `gemini-flash-latest` fallback) for vocabulary generation
- ARASAAC public API for AAC symbol mapping
- gTTS for text-to-speech (disk-cached MP3 per word, Web Speech API fallback)

## Evaluation: what RAG and child history actually do
`scripts/evaluate_rag_consistency.py` generates each board 3 times per photo
under three conditions (same photo and caption throughout) on 14 photos —
12 from COCO val2017 plus 2 of our own — 126 Gemini calls in total. Full
results: `data/evaluation/rag_consistency.json`.

| Condition | Consistency across runs (Jaccard) | Words from curated AAC sets |
|---|---|---|
| No RAG | 0.731 | 59.1% |
| RAG | 0.724 | 64.9% |
| RAG + child history | **0.960** | 65.8% |

- **RAG does not make wording more consistent** — once Gemini sees the
  photo, repeated runs are already fairly stable. What it does is steer
  vocabulary toward the curated, child-appropriate AAC word sets (+5.8
  points).
- **Child history is what delivers consistency**: reusing a child's prior
  words raises cross-run overlap from 0.72 to 0.96, which is what AAC motor
  planning needs (the same concept phrased the same way every time).

Limitations: 14 photos, 3 runs each, mostly general COCO scenes rather than
child-specific ones; "words from curated sets" measures alignment, not
clinical quality.

```bash
python scripts/evaluate_rag_consistency.py path/to/photos --runs 3 --out report.json
```

## Setup
```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

You need a [Gemini API key](https://aistudio.google.com/apikey). The normal
setup — and how a deployed instance runs — is a **server-side key**: copy
`.env.example` to `.env` and set `GEMINI_API_KEY` there. Caregivers then
never see a key field; they get a daily allowance of boards instead.

- `DAILY_BOARD_LIMIT` (default 30) caps how many boards per day (UTC) can
  use the server's key, counted in SQLite so restarts don't reset it. The
  create page shows how many are left; once it's used up, the user sees
  "Today's limit of N boards has been reached".
- If Google itself rejects the key for quota, the user sees "The Gemini API
  key has reached its usage limit for now"; an invalid key gives a clear
  "key isn't valid" message instead of a raw error.

With no server key, the UI shows a "Gemini API key" field instead — saved
in your browser's local storage only, and not counted against the limit
(it's your own quota).

## Run
```bash
python run.py
```

(Don't run plain `uvicorn app.main:app --reload` — the app writes to
`data/` on every board generation, which makes uvicorn's default
auto-reloader restart the whole server mid-request, and the browser just
shows "Failed to fetch" with no explanation. `run.py` excludes `data/`
from the reload watch so this doesn't happen.)

## Try it
```bash
curl -X POST http://127.0.0.1:8000/generate-board ^
  -F "photo=@sample.jpg" ^
  -F "scenario=practicing at the playground"
```

Or open http://127.0.0.1:8000/docs for the interactive Swagger UI.
