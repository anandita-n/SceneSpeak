"""Does RAG grounding actually help? Measures it instead of assuming it.

For each photo, generates the board vocabulary several times under three
conditions — without RAG, with the RAG word list, and with RAG plus the
child's history (same photo and caption throughout) — then reports:

- consistency: mean pairwise Jaccard overlap of the generated word sets
  across repeated runs of the same photo. AAC relies on stable wording
  (motor planning), so higher is better.
- corpus alignment: share of generated words that come from the curated
  AAC word sets (data/aac_scenario_vocabulary.json).

Usage:
    python scripts/evaluate_rag_consistency.py IMAGE_DIR [--runs 3] [--out report.json]

Needs GEMINI_API_KEY in .env. Makes 3 * runs Gemini calls per image.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.captioning import generate_caption  # noqa: E402
from app.rag import _load_corpus, retrieve_vocabulary_context  # noqa: E402
from app.vocab import FITZGERALD_ORDER, GeminiBusyError, GeminiQuotaError, generate_vocabulary  # noqa: E402

PAUSE_SECONDS = 4  # stay under the free tier's requests-per-minute limit


def word_set(vocab) -> set[str]:
    return {w.strip().lower() for c in FITZGERALD_ORDER for w in getattr(vocab, c)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def generate_with_retry(**kwargs):
    for attempt in range(4):
        try:
            return generate_vocabulary(**kwargs)
        except (GeminiBusyError, GeminiQuotaError) as exc:
            wait = 30 * (attempt + 1)
            print(f"    {type(exc).__name__}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("Gemini kept failing; stopping the evaluation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    api_key = dotenv_values(Path(__file__).resolve().parent.parent / ".env").get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY in .env first.")

    corpus_words = {e["word"].lower() for e in _load_corpus()}
    images = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    results = []

    for path in images:
        image = Image.open(path).convert("RGB")
        caption = generate_caption(image)
        retrieved = retrieve_vocabulary_context(caption, None)
        print(f"{path.name}: {caption}", flush=True)

        row = {"image": path.name, "caption": caption}
        # with_rag_and_history: the first run has no history; later runs get
        # the first run's words as the child's history — what happens when a
        # caregiver makes another board of the same kind of scene.
        for condition, context, use_history in (("without_rag", None, False), ("with_rag", retrieved, False),
                                                ("with_rag_and_history", retrieved, True)):
            runs = []
            for _ in range(args.runs):
                history = sorted(runs[0]) if use_history and runs else None
                vocab = generate_with_retry(caption=caption, image=image, retrieved_vocabulary=context,
                                            child_history=history, api_key=api_key)
                runs.append(word_set(vocab))
                time.sleep(PAUSE_SECONDS)
            consistency = sum(jaccard(a, b) for a, b in itertools.combinations(runs, 2)) / max(1, len(runs) * (len(runs) - 1) // 2)
            all_words = [w for r in runs for w in r]
            alignment = sum(w in corpus_words for w in all_words) / max(1, len(all_words))
            row[condition] = {"consistency": round(consistency, 3), "corpus_alignment": round(alignment, 3),
                              "runs": [sorted(r) for r in runs]}
            print(f"    {condition:22} consistency={consistency:.3f} corpus_alignment={alignment:.3f}", flush=True)
        results.append(row)

    summary = {}
    for condition in ("without_rag", "with_rag", "with_rag_and_history"):
        summary[condition] = {
            metric: round(sum(r[condition][metric] for r in results) / len(results), 3)
            for metric in ("consistency", "corpus_alignment")
        }
    print("\nMEAN over", len(results), "images:")
    for condition, metrics in summary.items():
        print(f"  {condition:22} consistency={metrics['consistency']:.3f} corpus_alignment={metrics['corpus_alignment']:.3f}")

    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "per_image": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
