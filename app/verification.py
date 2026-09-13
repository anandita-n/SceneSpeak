"""Phase 4: vision-language verification layer.

Generated vocabulary can name objects that aren't actually in the photo
(an LLM hallucinating from the caption, or a caption that missed
something). This layer independently scores each generated *object* word
against the real image using CLIP, and filters out words whose visual
similarity falls below a threshold — catching mistakes the caption-only
pipeline (Phases 1-3) has no way to notice on its own.

Scoped to the "objects" category deliberately: core vocabulary (want,
help, stop) and prepositions (in, up) aren't things a photo depicts, so
scoring them against image similarity would be meaningless — CLIP
verification only makes sense for words that claim something concrete is
visible in the scene.
"""

from functools import lru_cache

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
DEFAULT_THRESHOLD = 0.23  # measured: 90.9% precision / 83.3% recall / 85.7% accuracy on the 21-pair
                          # labeled test set in data/verification_testset (see scripts/evaluate_verification.py)


@lru_cache(maxsize=1)
def _load_model():
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


@torch.no_grad()
def clip_similarity(image: Image.Image, word: str) -> float:
    """Cosine similarity between the image and the text prompt "a photo of
    a {word}", scaled to roughly [0, 1] (CLIP's raw cosine sim for
    plausible image/text pairs typically falls in ~0.15-0.35)."""
    processor, model = _load_model()
    inputs = processor(text=[f"a photo of a {word}"], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    similarity = (image_embeds @ text_embeds.T).item()
    return similarity


def verify_objects(image: Image.Image, words: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Score each object word against the image. Returns one dict per word
    with its similarity score and whether it passed the threshold."""
    results = []
    for word in words:
        score = clip_similarity(image, word)
        results.append({"word": word, "score": round(score, 4), "passed": score >= threshold})
    return results


def filter_verified_objects(image: Image.Image, words: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    return [r["word"] for r in verify_objects(image, words, threshold) if r["passed"]]
