"""Captioning evaluation on the SceneSpeak eval images (COCO val2017).

Compares captioners against COCO's five human reference captions per image:

  blip_base             Salesforce/blip-image-captioning-base, 5 beams
  blip_base_scenespeak  the same model with SceneSpeak's decoding (3 beams,
                        no_repeat_ngram_size=3, repetition_penalty=1.3)
  blip_finetuned        BLIP fine-tuned in the GenAi image-captioning project
                        (Flickr8k), 5 beams
  gemini                Gemini, asked for a one-sentence caption, temperature 0

Metrics mirror the GenAi project's src/evaluate.py so the numbers are
comparable: corpus BLEU-4 (nltk, smoothing method1), mean METEOR (nltk),
mean ROUGE-L F1 (best reference), CIDEr-D (pycocoevalcap). Text is
lower-cased and punctuation is stripped before scoring.

Usage
  python eval/caption_eval.py --captions PATH/captions_val2017.json
      [--finetuned-dir PATH] [--models blip_base,blip_finetuned,gemini] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values  # noqa: E402
from PIL import Image  # noqa: E402

DATASET = ROOT / "eval" / "dataset"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_MD = ROOT / "RESULTS.md"
BASE_MODEL = "Salesforce/blip-image-captioning-base"
DEFAULT_FINETUNED = Path.home() / "GenAi" / "checkpoints" / "blip-finetuned"
GEMINI_MODELS = ("gemini-flash-lite-latest", "gemini-flash-latest")
GEMINI_PROMPT = ("Write a caption for this photo: one short, plain sentence (about 8-12 words) "
                 "describing the main subject and what is happening, like a typical photo caption. "
                 "Reply with the caption only.")
ALL_MODELS = ("blip_base", "blip_base_scenespeak", "blip_finetuned", "gemini")


# ---------------------------------------------------------------- captioners
def blip_captioner(model_dir: str, **generate_kwargs):
    """Return a caption(image) function for a BLIP checkpoint."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    try:
        processor = BlipProcessor.from_pretrained(model_dir)
    except Exception:
        # Fine-tuned checkpoints may not carry the image preprocessor config;
        # preprocessing is identical to the base model's.
        processor = BlipProcessor.from_pretrained(BASE_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(model_dir)
    # BLIP ties the text decoder's output layer to its word embeddings
    # (tie_word_embeddings=True). Checkpoints saved by transformers 5.x store
    # the tied weight only once, and 4.44 doesn't re-tie it on load — so it
    # would be left randomly initialised and the model would output
    # gibberish. Re-tie it explicitly (a no-op for the base model, where the
    # two are already equal).
    if model.config.text_config.tie_word_embeddings:
        model.text_decoder.cls.predictions.decoder.weight = model.text_decoder.bert.embeddings.word_embeddings.weight
    model.eval()

    def caption(image: Image.Image) -> str:
        import torch

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs, **generate_kwargs)
        return processor.decode(out[0], skip_special_tokens=True).strip()

    return caption


def gemini_captioner(api_key: str):
    import google.generativeai as genai
    from google.api_core import exceptions as gx

    genai.configure(api_key=api_key)
    used: dict[str, int] = {}

    def caption(image: Image.Image) -> str:
        image = image.copy()
        image.thumbnail((1024, 1024))
        for attempt in range(4):
            for name in GEMINI_MODELS:
                try:
                    resp = genai.GenerativeModel(name).generate_content(
                        [GEMINI_PROMPT, image], generation_config={"temperature": 0},
                        request_options={"retry": None, "timeout": 30})
                    used[name] = used.get(name, 0) + 1
                    return resp.text.strip().strip('"')
                except (gx.ServiceUnavailable, gx.DeadlineExceeded, gx.ResourceExhausted, gx.InternalServerError):
                    continue
            time.sleep(30 * (attempt + 1))
        raise RuntimeError("Gemini kept failing")

    caption.models_used = used  # type: ignore[attr-defined]
    return caption


# ---------------------------------------------------------------- metrics
def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_l(pred: list[str], refs: list[list[str]]) -> float:
    """LCS-based ROUGE-L F1 against the best-matching reference."""
    best = 0.0
    for ref in refs:
        if not pred or not ref:
            continue
        dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
        for i in range(1, len(pred) + 1):
            for j in range(1, len(ref) + 1):
                dp[i][j] = dp[i - 1][j - 1] + 1 if pred[i - 1] == ref[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[-1][-1]
        if lcs:
            p, r = lcs / len(pred), lcs / len(ref)
            best = max(best, 2 * p * r / (p + r))
    return best


def caption_metrics(preds: list[str], refs: list[list[str]]) -> dict[str, float]:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from pycocoevalcap.cider.cider import Cider

    from app.verification import _wordnet

    _wordnet()  # makes sure WordNet (needed by METEOR) is on nltk's path
    pred_tok = [tokens(p) for p in preds]
    refs_tok = [[tokens(r) for r in group] for group in refs]
    cider, _ = Cider().compute_score(
        {i: [" ".join(t) for t in group] for i, group in enumerate(refs_tok)},
        {i: [" ".join(t)] for i, t in enumerate(pred_tok)},
    )
    n = max(len(preds), 1)
    return {
        "BLEU-4": corpus_bleu(refs_tok, pred_tok, weights=(0.25,) * 4, smoothing_function=SmoothingFunction().method1),
        "METEOR": sum(meteor_score(g, p) for g, p in zip(refs_tok, pred_tok)) / n,
        "ROUGE-L": sum(rouge_l(p, g) for p, g in zip(pred_tok, refs_tok)) / n,
        "CIDEr": cider,
    }


# ---------------------------------------------------------------- main
def coco_id(label: dict) -> int | None:
    m = re.search(r"coco(\d+)", label["image"])
    return int(m.group(1)) if m else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captions", type=Path, required=True, help="COCO captions_val2017.json")
    parser.add_argument("--finetuned-dir", type=Path, default=DEFAULT_FINETUNED)
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    refs_by_id: dict[int, list[str]] = {}
    for ann in json.loads(args.captions.read_text(encoding="utf-8"))["annotations"]:
        refs_by_id.setdefault(ann["image_id"], []).append(ann["caption"].strip())

    labels = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((DATASET / "labels").glob("*.json"))]
    items = [(lab, coco_id(lab)) for lab in labels]
    items = [(lab, i) for lab, i in items if i in refs_by_id and (DATASET / "images" / lab["image"]).exists()]
    if args.limit:
        items = items[: args.limit]
    if not items:
        sys.exit("No COCO-sourced eval images with reference captions found.")

    captioners = {}
    if "blip_base" in models:
        captioners["blip_base"] = blip_captioner(BASE_MODEL, num_beams=5, max_new_tokens=40)
    if "blip_base_scenespeak" in models:
        captioners["blip_base_scenespeak"] = blip_captioner(
            BASE_MODEL, num_beams=3, max_new_tokens=30, no_repeat_ngram_size=3, repetition_penalty=1.3)
    if "blip_finetuned" in models:
        if not args.finetuned_dir.exists():
            sys.exit(f"Fine-tuned checkpoint not found: {args.finetuned_dir}")
        captioners["blip_finetuned"] = blip_captioner(str(args.finetuned_dir), num_beams=5, max_new_tokens=40)
    if "gemini" in models:
        key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY")
        if not key:
            sys.exit("Set GEMINI_API_KEY in .env for the gemini captioner.")
        captioners["gemini"] = gemini_captioner(key)

    started = datetime.now()
    out_dir = RESULTS_DIR / (started.strftime("%Y-%m-%d_%H%M") + "_captions")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for n, (label, img_id) in enumerate(items, start=1):
        image = Image.open(DATASET / "images" / label["image"]).convert("RGB")
        rec = {"image": label["image"], "coco_id": img_id, "references": refs_by_id[img_id]}
        for name, fn in captioners.items():
            t = time.perf_counter()
            rec[name] = fn(image)
            rec[f"{name}_ms"] = round((time.perf_counter() - t) * 1000)
            if name == "gemini":
                time.sleep(4)  # free-tier rate limit; outside the timed section
        records.append(rec)
        print(f"[{n}/{len(items)}] {label['image']}", *(f"  {m:22} {rec[m]}" for m in captioners), sep="\n", flush=True)

    with (out_dir / "captions.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    refs = [r["references"] for r in records]
    summary = []
    for name in captioners:
        scores = caption_metrics([r[name] for r in records], refs)
        times = sorted(r[f"{name}_ms"] for r in records)
        summary.append({"captioner": name, **scores, "median_ms": times[len(times) // 2],
                        "mean_words": sum(len(tokens(r[name])) for r in records) / len(records)})
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    gemini_models = getattr(captioners.get("gemini"), "models_used", None)
    lines = [
        f"## Captioning evaluation {started:%Y-%m-%d %H:%M}",
        "",
        f"- Command: `python eval/caption_eval.py {' '.join(sys.argv[1:])}`",
        f"- Git commit: `{commit}` · images: {len(records)} COCO val2017 eval images · references: COCO's 5 human captions each",
        f"- Fine-tuned model: `{args.finetuned_dir}` (GenAi project; config: 400 Flickr8k training images, 3 epochs, CPU)"
        if "blip_finetuned" in captioners else "- Fine-tuned model: not run",
        f"- Gemini models that answered: {gemini_models}" if gemini_models is not None else "- Gemini: not run",
        f"- Raw output: `{out_dir.relative_to(ROOT).as_posix()}/` (captions.jsonl, summary.csv)",
        "",
        "| Captioner | BLEU-4 | METEOR | ROUGE-L | CIDEr | Words/caption | Median time |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(f"| {s['captioner']} | {s['BLEU-4']:.4f} | {s['METEOR']:.4f} | {s['ROUGE-L']:.4f} | "
                     f"{s['CIDEr']:.4f} | {s['mean_words']:.1f} | {s['median_ms'] / 1000:.2f}s |")
    lines += ["", "Example captions (first 3 images):", ""]
    for rec in records[:3]:
        lines.append(f"- `{rec['image']}` — reference: “{rec['references'][0]}”")
        for name in captioners:
            lines.append(f"  - {name}: “{rec[name]}”")
    lines += ["", "Metrics mirror the GenAi project's `src/evaluate.py` (lower-cased, punctuation stripped). "
              "Gemini was prompted for a one-sentence caption at temperature 0; its time is the API round trip "
              "(BLIP times are local CPU inference).", ""]
    block = "\n".join(lines)
    print("\n" + block)
    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write("\n" + block)
    (out_dir / "summary.md").write_text(block, encoding="utf-8")


if __name__ == "__main__":
    main()
