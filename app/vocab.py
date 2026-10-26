"""Phase 1: multimodal vocabulary generation from a scene photo.

Follows the QuickPic AAC approach (Yu et al., IJERPH 2024): the LLM
generates topic-specific vocabulary per part of speech, and the board lays
those categories out in modified Fitzgerald Key order — pronouns, verbs,
prepositions, descriptors, objects, left to right — so symbols sit in the
grammatical position a child would reach for them in a sentence. A
child's core vocabulary lives on their main AAC device, so a topic board
doesn't try to regenerate it.

Gemini receives the photo itself, not just BLIP's caption (as in
Zastudil et al. 2024, who gave GPT-4V the image directly). Working from a
caption alone, a language model has to guess which objects a scene
"probably" contains — the co-occurrence hallucination POPE (Li et al.,
EMNLP 2023) documents — and an earlier CLIP-similarity filter meant to
catch those guesses rejected real, visible objects on real photos. Seeing
the image removes the guesswork at the source; the caregiver's Edit mode is
the remaining safety net, as in QuickPic. BLIP's caption still drives RAG
retrieval and is shown to the caregiver as the scene description.

Phase 2 adds RAG grounding (scenario word sets via ChromaDB + per-child
history) on top of this same function; Phase 3 adds symbol mapping.
"""

import json
import os
from functools import lru_cache

import google.generativeai as genai
from PIL import Image
from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel, Field

# Google's "-latest" aliases always point at the current Flash models, so
# they don't go stale the way pinned versions do (gemini-1.5-flash was
# retired, and gemini-2.5-flash has no free-tier quota for newly created
# keys — both broke this app). Lite goes first: picking simple vocabulary is
# a small task (QuickPic used GPT-3.5), and Lite answered in ~3s where full
# Flash was frequently overloaded. If every alias 404s, we fall back to
# auto-discovering a model the caller's key can actually use.
_PREFERRED_MODEL_NAMES = ("gemini-flash-lite-latest", "gemini-flash-latest")

# The client library's default retry policy silently retries 503/429s with
# backoff for minutes (measured: ~3 min for one overloaded call). Fail fast
# instead and move to the next model.
_REQUEST_OPTIONS = {"retry": None, "timeout": 30}
# JSON mode: the model returns a bare JSON object, so no markdown-fence
# stripping or "only JSON please" prompt tricks are needed on the happy path.
_GENERATION_CONFIG = {"response_mime_type": "application/json"}
# Gemini doesn't need full camera resolution to recognise a scene, and a
# smaller upload is faster.
_MAX_IMAGE_SIDE = 1024
_TRY_NEXT_MODEL = (
    google_exceptions.NotFound,
    google_exceptions.ServiceUnavailable,
    google_exceptions.DeadlineExceeded,
    google_exceptions.InternalServerError,
    google_exceptions.ResourceExhausted,
)


# Modified Fitzgerald Key column order, left to right, as used by QuickPic.
FITZGERALD_ORDER = ("pronouns", "verbs", "prepositions", "descriptors", "objects")


class VocabularySet(BaseModel):
    pronouns: list[str] = Field(default_factory=list, description="People/pronouns a child would use about this scene (I, you, we, my...)")
    verbs: list[str] = Field(default_factory=list, description="Actions in this scene, base form (play, push, eat...)")
    prepositions: list[str] = Field(default_factory=list, description="Spatial/relational words relevant to the scene")
    descriptors: list[str] = Field(default_factory=list, description="Adjectives/adverbs relevant to the scene")
    objects: list[str] = Field(default_factory=list, description="Nouns specific to this scene")


@lru_cache(maxsize=8)
def _discover_model_name(api_key: str) -> str:
    """Ask this API key which models it can actually use, and pick a
    generateContent-capable one — preferring a "flash" model (cheaper/
    faster) if one is available."""
    genai.configure(api_key=api_key)
    available = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    if not available:
        raise RuntimeError(
            "This Gemini API key doesn't have access to any model that supports "
            "generateContent. Check the key at https://aistudio.google.com/apikey."
        )
    # Skip specialised variants (image/TTS/preview) — they either can't do
    # plain text generation well or disappear without notice.
    general = [m for m in available if not any(t in m.lower() for t in ("image", "tts", "preview", "omni"))]
    flash_models = [m for m in general if "flash" in m.lower()]
    return (flash_models or general or available)[0]


def _resolve_api_key(api_key: str | None) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Either paste one into the API key field in the "
            "UI, or copy .env.example to .env and set GEMINI_API_KEY there."
        )
    return api_key


class GeminiQuotaError(RuntimeError):
    """The key has run out of Gemini quota (rate limit or daily cap)."""


class GeminiKeyError(RuntimeError):
    """The key is invalid, revoked, or not allowed to use the API."""


class GeminiBusyError(RuntimeError):
    """Every model we tried was overloaded or timed out."""


def _generate(contents, api_key: str):
    """Try the preferred models in order, moving on when one is retired
    (404), overloaded (503), timed out, or out of quota for this key —
    a construction-time GenerativeModel() call never hits the API, so those
    only surface on generate_content(). Quota, busy, and bad-key failures are
    translated into messages a caregiver can act on, instead of a raw
    Google error."""
    genai.configure(api_key=api_key)
    errors = []
    try:
        for model_name in _PREFERRED_MODEL_NAMES:
            try:
                return genai.GenerativeModel(model_name).generate_content(
                    contents, generation_config=_GENERATION_CONFIG, request_options=_REQUEST_OPTIONS)
            except _TRY_NEXT_MODEL as exc:
                errors.append(exc)
        if all(isinstance(e, google_exceptions.NotFound) for e in errors):
            model_name = _discover_model_name(api_key)
            return genai.GenerativeModel(model_name).generate_content(
                contents, generation_config=_GENERATION_CONFIG, request_options=_REQUEST_OPTIONS)
        quota_errors = [e for e in errors if isinstance(e, google_exceptions.ResourceExhausted)]
        if quota_errors:
            raise quota_errors[0]
        raise GeminiBusyError("Gemini is very busy right now. Please try again in a minute.") from errors[-1]
    except google_exceptions.ResourceExhausted as exc:
        raise GeminiQuotaError(
            "The Gemini API key has reached its usage limit for now. "
            "Please try again later (Google's free-tier quota resets daily)."
        ) from exc
    except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated) as exc:
        raise GeminiKeyError("The Gemini API key was rejected. Check that it's valid and enabled.") from exc
    except google_exceptions.InvalidArgument as exc:
        if "api key" in str(exc).lower():
            raise GeminiKeyError("The Gemini API key isn't valid. Check it at https://aistudio.google.com/apikey.") from exc
        raise


def _build_prompt(
    caption: str,
    scenario: str | None,
    retrieved_vocabulary: list[dict] | None = None,
    child_history: list[str] | None = None,
) -> str:
    context = f'Automatic caption of the photo (may miss details): "{caption}"'
    if scenario:
        context += f'\nActivity/context provided by the caregiver: "{scenario}"'

    grounding = ""
    if retrieved_vocabulary:
        word_list = ", ".join(sorted({e["word"] for e in retrieved_vocabulary}))
        grounding += (
            "\n\nPrefer words from these curated AAC scenario word sets when they fit "
            f"the scene (only include ones that are actually relevant, don't force all of them): {word_list}"
        )
    if child_history:
        grounding += (
            "\n\nThis child has previously used these words on other boards — reuse the "
            f"same wording for any that fit this scene, instead of inventing new phrasing "
            f"for the same concept: {', '.join(child_history)}"
        )

    return f"""You support a speech-language pathologist building an AAC
(Augmentative and Alternative Communication) topic display for a nonverbal
or minimally-verbal child. Given the scene below, suggest vocabulary the
child could use to talk about it during a language-learning activity. The
photo of the scene is attached — base your suggestions on what you can
actually see in it. The display will be arranged by part of speech in
Fitzgerald Key order.

{context}{grounding}

Return words for each category:
- pronouns: 3-6 words (e.g. I, you, we, my, it)
- verbs: 6-10 actions, in base form (e.g. play, push, eat)
- prepositions: 4-8 spatial/relational words
- descriptors: 6-10 adjectives/adverbs
- objects: 6-10 nouns for things you can actually see in the photo (never
  objects that are merely typical of this kind of scene but not shown)

Each word must truly be that category's part of speech — e.g. "ready" and
"finished" are descriptors, not verbs; social phrases like "my turn" belong
in none of these lists. Never repeat a word across categories. Use simple,
literal words a child could say in the first person; a short verb phrase
(e.g. "raise hand") is fine where one word won't do. Avoid rare or abstract
vocabulary, and only suggest words relevant to this scene.

Respond with a JSON object shaped exactly like this:
{{"pronouns": [...], "verbs": [...], "prepositions": [...], "descriptors": [...], "objects": [...]}}"""


def generate_vocabulary(
    caption: str,
    scenario: str | None = None,
    image: Image.Image | None = None,
    retrieved_vocabulary: list[dict] | None = None,
    child_history: list[str] | None = None,
    api_key: str | None = None,
) -> VocabularySet:
    resolved_key = _resolve_api_key(api_key)
    prompt = _build_prompt(caption, scenario, retrieved_vocabulary, child_history)
    contents = [prompt]
    if image is not None:
        image = image.copy()
        image.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
        contents.append(image)
    # Even in JSON mode the model occasionally returns malformed JSON
    # (observed ~1 in 50 calls during evaluation); one retry is far cheaper
    # than failing the caregiver's whole board.
    for attempt in range(2):
        response = _generate(contents, resolved_key)
        try:
            return VocabularySet(**json.loads(_strip_fences(response.text)))
        except (json.JSONDecodeError, ValueError, TypeError):
            if attempt == 1:
                raise


def _strip_fences(text: str) -> str:
    # JSON mode should make this unnecessary; kept as a fallback for a
    # discovered model that ignores response_mime_type.
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    return text
