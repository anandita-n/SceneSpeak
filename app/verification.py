"""Lightweight, pluggable verification: does a suggested word match the photo?

Gemini already sees the photo, so this is a second opinion, not a gate:
words are *scored*, and low-scoring ones are shown greyed out on the board
rather than silently dropped — the caregiver decides. The earlier CLIP filter
deleted words outright and, on real photos, deleted visible objects (desk,
chair, pencil in a classroom), which is why this version never removes words.

Verifiers (choose with config/verification.json or the VERIFIER env var):
- "none": no scoring (baseline).
- "caption_overlap": no extra model. A word passes if it, its lemma or a
  WordNet synonym appears in BLIP captions of the photo (1 beam + N sampled).
- "blip_itm": BLIP image-text matching (Salesforce/blip-itm-base-coco), the
  same model family as the captioner. Scores "a photo containing {word}" for
  every word in one batched forward pass.

Only categories in `verify_categories` are scored, and protected core words
(I, want, more, stop, ...) are never scored at all.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "verification.json"
NLTK_DATA_DIR = ROOT / "data" / "nltk_data"

logger = logging.getLogger("scenespeak.verification")


@dataclass(frozen=True)
class WordScore:
    """Verification result for one word. `score` is None when the word wasn't
    scored (protected, outside the verified categories, or verifier "none")."""

    word: str
    score: float | None
    verified: bool
    protected: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Verifier(Protocol):
    """Anything that can score words against an image."""

    name: str

    def verify(self, image: Image.Image, words: list[str], caption: str | None = None) -> list[WordScore]:
        """Score each word against the image, in the same order as `words`."""
        ...


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def protected_words() -> set[str]:
    return {w.lower() for w in load_config()["protected_words"]}


# ---------------------------------------------------------------- none
class NoVerifier:
    """Baseline: every word passes, nothing is scored."""

    name = "none"

    def verify(self, image: Image.Image, words: list[str], caption: str | None = None) -> list[WordScore]:
        return [WordScore(word=w, score=None, verified=True) for w in words]


# ---------------------------------------------------------------- caption_overlap
@lru_cache(maxsize=1)
def _wordnet():
    """Load WordNet, downloading it into data/nltk_data on first use."""
    import nltk

    NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DATA_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA_DIR))
    for resource in ("wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer

    return wordnet, WordNetLemmatizer()


def _lemmas(token: str) -> set[str]:
    _, lemmatizer = _wordnet()
    return {token, lemmatizer.lemmatize(token, "n"), lemmatizer.lemmatize(token, "v")}


def _word_forms(word: str, max_synsets: int = 2) -> set[str]:
    """The word, its lemmas, and synonyms from its first few WordNet senses
    (kept small: deeper senses add far-fetched matches like ball -> globe)."""
    wordnet, _ = _wordnet()
    word = word.lower().strip()
    forms = {word} if " " in word else _lemmas(word)
    key = word.replace(" ", "_")
    for pos in (wordnet.NOUN, wordnet.VERB):
        for synset in wordnet.synsets(key, pos=pos)[:max_synsets]:
            forms.update(name.replace("_", " ").lower() for name in synset.lemma_names())
    return forms


def _caption_terms(caption: str) -> tuple[set[str], str]:
    """Single-word lemmas in the caption, plus the normalised text for
    matching multi-word phrases like "ice cream"."""
    tokens = re.findall(r"[a-z]+", caption.lower())
    terms: set[str] = set()
    for token in tokens:
        terms |= _lemmas(token)
    return terms, " " + " ".join(tokens) + " "


class CaptionOverlapVerifier:
    """Score = fraction of captions (1 beam + N sampled) that mention the word
    or a synonym. No extra model: reuses the BLIP captioner."""

    name = "caption_overlap"

    def __init__(self, threshold: float, samples: int) -> None:
        self.threshold = threshold
        self.samples = samples

    def verify(self, image: Image.Image, words: list[str], caption: str | None = None) -> list[WordScore]:
        from app.captioning import generate_caption, generate_sampled_captions

        if not words:
            return []
        captions = [caption or generate_caption(image)] + generate_sampled_captions(image, self.samples)
        parsed = [_caption_terms(c) for c in captions]
        results = []
        for word in words:
            forms = _word_forms(word)
            hits = sum(
                1 for terms, text in parsed
                if any((f" {form} " in text) if " " in form else (form in terms) for form in forms)
            )
            score = hits / len(captions)
            results.append(WordScore(word=word, score=round(score, 3), verified=score >= self.threshold))
        return results


# ---------------------------------------------------------------- blip_itm
@lru_cache(maxsize=1)
def _load_itm(model_name: str):
    from transformers import BlipForImageTextRetrieval, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForImageTextRetrieval.from_pretrained(model_name)
    model.eval()
    return processor, model


class BlipItmVerifier:
    """BLIP image-text matching. The image is encoded once and every word's
    prompt is scored against it in a single batched pass through the text
    encoder + ITM head (the same computation BlipForImageTextRetrieval does
    per pair, without re-encoding the image for each word)."""

    name = "blip_itm"

    def __init__(self, model_name: str, prompt: str, threshold: float) -> None:
        self.model_name = model_name
        self.prompt = prompt
        self.threshold = threshold

    def verify(self, image: Image.Image, words: list[str], caption: str | None = None) -> list[WordScore]:
        import torch

        if not words:
            return []
        processor, model = _load_itm(self.model_name)
        with torch.no_grad():
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            image_embeds = model.vision_model(pixel_values=pixel_values)[0]
            image_embeds = image_embeds.expand(len(words), -1, -1)
            image_atts = torch.ones(image_embeds.shape[:-1], dtype=torch.long)
            text = processor.tokenizer(
                [self.prompt.format(word=w) for w in words], padding=True, return_tensors="pt"
            )
            hidden = model.text_encoder(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
            )[0]
            probs = torch.softmax(model.itm_head(hidden[:, 0, :]), dim=-1)[:, 1].tolist()
        return [
            WordScore(word=w, score=round(p, 3), verified=p >= self.threshold)
            for w, p in zip(words, probs)
        ]


# ---------------------------------------------------------------- selection
VERIFIER_NAMES = ("none", "caption_overlap", "blip_itm")


def get_verifier(name: str | None = None) -> Verifier:
    """Build the verifier named by `name`, the VERIFIER env var, or the
    config file — in that order of precedence."""
    config = load_config()
    name = (name or os.environ.get("VERIFIER") or config["verifier"]).strip().lower()
    thresholds = config["thresholds"]
    if name == "none":
        return NoVerifier()
    if name == "caption_overlap":
        return CaptionOverlapVerifier(thresholds["caption_overlap"], config["caption_overlap_samples"])
    if name == "blip_itm":
        return BlipItmVerifier(config["blip_itm_model"], config["blip_itm_prompt"], thresholds["blip_itm"])
    raise ValueError(f"Unknown verifier {name!r}; expected one of {VERIFIER_NAMES}")


def verify_board(
    image: Image.Image,
    vocab: dict[str, list[str]],
    caption: str | None = None,
    verifier: Verifier | None = None,
) -> tuple[dict[str, WordScore], float]:
    """Score a generated board's words. Returns {word: WordScore} for every
    word on the board, and the verifier's latency in milliseconds.

    Protected words and categories outside `verify_categories` are marked
    as not scored; everything else is sent to the verifier in one batch.
    """
    verifier = verifier or get_verifier()
    protected = protected_words()
    categories = set(load_config()["verify_categories"])

    results: dict[str, WordScore] = {}
    to_check: list[str] = []
    for category, words in vocab.items():
        for word in words:
            if word in results or word in to_check:
                continue
            if word.lower() in protected:
                results[word] = WordScore(word=word, score=None, verified=True, protected=True)
            elif category not in categories:
                results[word] = WordScore(word=word, score=None, verified=True)
            else:
                to_check.append(word)

    start = time.perf_counter()
    for scored in verifier.verify(image, to_check, caption=caption):
        results[scored.word] = scored
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "verifier=%s checked=%d low_confidence=%d latency_ms=%.0f",
        verifier.name, len(to_check),
        sum(1 for w in to_check if not results[w].verified), latency_ms,
    )
    return results, latency_ms
