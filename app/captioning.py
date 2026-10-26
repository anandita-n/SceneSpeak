"""BLIP-based image captioning — the vision-language component reused
from the original image-captioning project. Its caption is the query for
RAG retrieval (app/rag.py), goes into the generation prompt alongside the
photo, and is shown to the caregiver as the scene description."""

from functools import lru_cache

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
    illustration of a girl eating an ice cream stock stock stock ...".
    """
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=3,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
    )
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()
