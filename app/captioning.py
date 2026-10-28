"""BLIP-based image captioning — the vision-language component reused
from the original image-captioning project. Its caption is the query for
RAG retrieval (app/rag.py), goes into the generation prompt alongside the
photo, and is shown to the caregiver as the scene description."""

from functools import lru_cache

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def _load_model():
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def generate_caption(image: Image.Image, max_new_tokens: int = 30) -> str:
    """Generate a natural-language caption describing the scene in `image`.

    no_repeat_ngram_size stops beam search looping on a phrase — without it
    a stock-image style photo came back as "a girl eating an ice cream stock
    illustration of a girl eating an ice cream stock stock stock ...". A
    repetition_penalty used to be set too; the caption evaluation showed it
    made no difference (CIDEr 1.277 with it, 1.288 without, 50 COCO images),
    so it was dropped. Greedy decoding was 2.4x faster but scored lower
    (CIDEr 1.211), so beam search stays.
    """
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=3,
        no_repeat_ngram_size=3,
    )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def generate_sampled_captions(image: Image.Image, n: int = 2, seed: int = 0, max_new_tokens: int = 30) -> list[str]:
    """Generate `n` extra captions with nucleus sampling, for broader coverage
    of what's in the scene than the single beam-search caption gives. Seeded,
    so the same image always yields the same captions (repeatable eval)."""
    if n <= 0:
        return []
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    torch.manual_seed(seed)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=n,
        no_repeat_ngram_size=3,
    )
    return [processor.decode(ids, skip_special_tokens=True).strip() for ids in output_ids]
