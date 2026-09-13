"""BLIP-based image captioning — the base vision-language component reused
from the original image-captioning project, now serving as the input
signal for vocabulary generation (and later, the verification layer)."""

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


def generate_caption(image: Image.Image, max_new_tokens: int = 40) -> str:
    """Generate a natural-language caption describing the scene in `image`."""
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=5)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()
