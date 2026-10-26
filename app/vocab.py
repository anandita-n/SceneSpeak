"""Phase 1: LLM-based vocabulary generation from a scene caption.

Follows the QuickPic AAC approach (Yu et al., IJERPH 2024): the LLM
generates topic-specific vocabulary per part of speech, and the board lays
those categories out in modified Fitzgerald Key order — pronouns, verbs,
prepositions, descriptors, objects, left to right — so symbols sit in the
grammatical position a child would reach for them in a sentence. A
child's core vocabulary lives on their main AAC device, so a topic board
doesn't try to regenerate it.

Phase 2 adds RAG grounding (scenario word sets via ChromaDB + per-child
history) on top of this same function; Phase 3 adds symbol mapping; Phase 4
adds the CLIP verification layer that checks objects against the photo.
"""

import json
import os
from functools import lru_cache

import google.generativeai as genai
from pydantic import BaseModel, Field

# Preferred model, tried first since it avoids an extra list_models() call on
# the common path. Google periodically retires/renames model versions, so if
# this 404s we fall back to auto-discovering whatever model the caller's own
# API key actually has access to, rather than hardcoding a name that can go
# stale (this already happened once during development).
_PREFERRED_MODEL_NAME = "gemini-1.5-flash"


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
    flash_models = [m for m in available if "flash" in m.lower()]
    return (flash_models or available)[0]


def _resolve_api_key(api_key: str | None) -> str:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Either paste one into the API key field in the "
            "UI, or copy .env.example to .env and set GEMINI_API_KEY there."
        )
    return api_key


def _generate(prompt: str, api_key: str):
    """Try the preferred model; a construction-time GenerativeModel() call
    never actually hits the API, so the 404 for a retired/unavailable model
    only surfaces on this generate_content() call — that's what we catch
    and fall back from."""
    genai.configure(api_key=api_key)
    try:
        return genai.GenerativeModel(_PREFERRED_MODEL_NAME).generate_content(prompt)
    except Exception:
        model_name = _discover_model_name(api_key)
        return genai.GenerativeModel(model_name).generate_content(prompt)


def _build_prompt(
    caption: str,
    scenario: str | None,
    retrieved_vocabulary: list[dict] | None = None,
    child_history: list[str] | None = None,
) -> str:
    context = f'Scene description: "{caption}"'
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
display will be arranged by part of speech in Fitzgerald Key order.

{context}{grounding}

Return words for each category:
- pronouns: 3-6 words (e.g. I, you, we, my, it)
- verbs: 6-10 actions, in base form (e.g. play, push, eat)
- prepositions: 4-8 spatial/relational words
- descriptors: 6-10 adjectives/adverbs
- objects: 6-10 nouns that are actually visible or present in the scene

Use simple, literal words a child could say in the first person; short
phrases (e.g. "my turn") are fine where one word won't do. Avoid rare or
abstract vocabulary, and only suggest words relevant to this scene.

Respond with only a JSON object shaped exactly like this, no other text:
{{"pronouns": [...], "verbs": [...], "prepositions": [...], "descriptors": [...], "objects": [...]}}"""


def generate_vocabulary(
    caption: str,
    scenario: str | None = None,
    retrieved_vocabulary: list[dict] | None = None,
    child_history: list[str] | None = None,
    api_key: str | None = None,
) -> VocabularySet:
    resolved_key = _resolve_api_key(api_key)
    prompt = _build_prompt(caption, scenario, retrieved_vocabulary, child_history)
    response = _generate(prompt, resolved_key)
    text = response.text.strip()
    # Strip accidental markdown code fences before parsing.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    return VocabularySet(**data)
