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
    a {word}". Single-word convenience wrapper — batches internally through
    clip_similarities, so scoring many words for the same image should use
    that instead (this re-encodes the image every call, which is wasteful
    when scoring more than one word)."""
    return clip_similarities(image, [word])[0]


@torch.no_grad()
def clip_similarities(image: Image.Image, words: list[str]) -> list[float]:
    """Cosine similarity between one image and each word, encoding the
    image only once and batching all the text prompts through CLIP in a
    single forward pass. The naive per-word version was re-running CLIP's
    image encoder (the expensive part) once per word — for an 8-word board
    that's 8x more image-encoding work than necessary for the exact same
    image. This is the actual fix for the "verification feels slow"
    complaint, not a smaller model."""
    if not words:
        return []
    processor, model = _load_model()

    image_inputs = processor(images=image, return_tensors="pt")
    image_embeds = model.get_image_features(**image_inputs)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    text_inputs = processor(text=[f"a photo of a {w}" for w in words], return_tensors="pt", padding=True)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    similarities = (text_embeds @ image_embeds.T).squeeze(-1)
    return similarities.tolist()


def verify_objects(image: Image.Image, words: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[dict]:
    """Score each object word against the image. Returns one dict per word
    with its similarity score and whether it passed the threshold."""
    scores = clip_similarities(image, words)
    return [
        {"word": word, "score": round(score, 4), "passed": score >= threshold}
        for word, score in zip(words, scores)
    ]


def filter_verified_objects(image: Image.Image, words: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[str]:
    return [r["word"] for r in verify_objects(image, words, threshold) if r["passed"]]
