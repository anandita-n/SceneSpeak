"""Evaluate SceneSpeak board generation against the labelled dataset.

Configurations
  A0  caption only — Gemini gets BLIP's caption, not the photo; no RAG, no verifier
  A   LLM only     — Gemini gets the photo + caption; no RAG, no verifier
  B   LLM + RAG
  C   LLM + RAG + caption_overlap verifier
  D   LLM + RAG + blip_itm verifier

C and D score the *same* boards B generated in that run: verification is
post-processing that never changes what Gemini returns, so re-generating
would only add API calls and noise. Their latency is B's plus the verifier.

Scene-specific words are the board's "objects" (minus protected core words).
For C/D only words the verifier keeps count; the app greys out the rest.

Every run uses temperature 0. Outputs go to eval/results/<timestamp>/ and a
summary is appended to RESULTS.md.

Usage
  python eval/run_eval.py --limit 5            # smoke test
  python eval/run_eval.py --runs 3             # full run
  python eval/run_eval.py --configs B,D --runs 1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import dotenv_values  # noqa: E402
from PIL import Image  # noqa: E402

from app.captioning import generate_caption  # noqa: E402
from app.rag import _load_corpus, retrieve_vocabulary_context  # noqa: E402
from app.verification import get_verifier, load_config, protected_words, verify_board  # noqa: E402
from app.vocab import FITZGERALD_ORDER, GeminiBusyError, GeminiQuotaError, generate_vocabulary  # noqa: E402
from eval.metrics import grounding_rate, matches, mean_of, percentile, score_board  # noqa: E402

DATASET = ROOT / "eval" / "dataset"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_MD = ROOT / "RESULTS.md"

CONFIGS: dict[str, dict] = {
    "A0": {"image": False, "rag": False, "verifier": None, "label": "Caption only (no photo), no RAG"},
    "A": {"image": True, "rag": False, "verifier": None, "label": "LLM with photo, no RAG"},
    "B": {"image": True, "rag": True, "verifier": None, "label": "LLM + RAG"},
    "C": {"image": True, "rag": True, "verifier": "caption_overlap", "label": "LLM + RAG + caption_overlap"},
    "D": {"image": True, "rag": True, "verifier": "blip_itm", "label": "LLM + RAG + blip_itm"},
}
GENERATING = ("A0", "A", "B")

SWEEP = {
    "caption_overlap": [0.0, 0.33, 0.66, 1.0],
    "blip_itm": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
}

PAUSE_SECONDS = 4  # keep under the free tier's requests-per-minute limit


class _ModelLog(logging.Handler):
    """Record which Gemini model actually answered each call."""

    def __init__(self) -> None:
        super().__init__()
        self.models: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if msg.startswith("gemini model="):
            self.models[msg.split("=", 1)[1]] += 1


def load_dataset(limit: int | None) -> list[dict]:
    """Labelled images that exist on disk, interleaved across scene types so
    a --limit sample covers several scenes rather than one."""
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for path in sorted((DATASET / "labels").glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        if (DATASET / "images" / label["image"]).exists():
            by_scene[label["scene"]].append(label)
    ordered: list[dict] = []
    while any(by_scene.values()):
        for scene in sorted(by_scene):
            if by_scene[scene]:
                ordered.append(by_scene[scene].pop(0))
    return ordered[:limit] if limit else ordered


def generate_with_retry(**kwargs):
    for attempt in range(4):
        try:
            return generate_vocabulary(**kwargs)
        except (GeminiBusyError, GeminiQuotaError) as exc:
            wait = 30 * (attempt + 1)
            print(f"      {type(exc).__name__}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("Gemini kept failing")


def scene_words(vocab: dict[str, list[str]], keep: set[str] | None = None) -> list[str]:
    protected = protected_words()
    return [w for w in vocab["objects"] if w.lower() not in protected and (keep is None or w in keep)]


def fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "–"
    return f"{value * 100:.1f}%" if pct else f"{value:.2f}"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="evaluate only the first N images (spread across scenes)")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--configs", default="A0,A,B,C,D")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(CONFIGS)
    if unknown:
        parser.error(f"unknown configs: {sorted(unknown)}")
    needed = [c for c in GENERATING if c in configs or (c == "B" and {"C", "D"} & set(configs))]

    api_key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY in .env first.")

    model_log = _ModelLog()
    logging.getLogger("scenespeak.vocab").addHandler(model_log)
    logging.getLogger("scenespeak.vocab").setLevel(logging.INFO)

    dataset = load_dataset(args.limit)
    if not dataset:
        sys.exit("No labelled images found in eval/dataset — see eval/dataset/README.md.")
    corpus = {e["word"].lower() for e in _load_corpus()}
    verifiers = {CONFIGS[c]["verifier"]: get_verifier(CONFIGS[c]["verifier"]) for c in configs if CONFIGS[c]["verifier"]}

    started = datetime.now()
    out_dir = RESULTS_DIR / started.strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = (out_dir / "raw.jsonl").open("w", encoding="utf-8")

    rows: list[dict] = []
    sweep_boards: dict[str, list[tuple[list[tuple[str, float | None]], dict]]] = defaultdict(list)

    for i, label in enumerate(dataset, start=1):
        image = Image.open(DATASET / "images" / label["image"]).convert("RGB")
        t = time.perf_counter(); caption = generate_caption(image); caption_ms = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); retrieved = retrieve_vocabulary_context(caption, None); rag_ms = (time.perf_counter() - t) * 1000
        print(f"[{i}/{len(dataset)}] {label['image']} — {caption}", flush=True)

        for run in range(1, args.runs + 1):
            boards: dict[str, tuple[dict, float]] = {}
            for name in needed:
                cfg = CONFIGS[name]
                t = time.perf_counter()
                vocab = generate_with_retry(
                    caption=caption, image=image if cfg["image"] else None,
                    retrieved_vocabulary=retrieved if cfg["rag"] else None,
                    child_history=None, api_key=api_key, temperature=0,
                ).model_dump()
                llm_ms = (time.perf_counter() - t) * 1000
                vocab = {c: vocab[c] for c in FITZGERALD_ORDER}
                boards[name] = (vocab, caption_ms + (rag_ms if cfg["rag"] else 0) + llm_ms)
                time.sleep(PAUSE_SECONDS)

            for name in configs:
                cfg = CONFIGS[name]
                source = name if name in GENERATING else "B"
                vocab, latency_ms = boards[source]
                keep = None
                verify_ms = 0.0
                if cfg["verifier"]:
                    scores, verify_ms = verify_board(image, vocab, caption, verifiers[cfg["verifier"]])
                    keep = {w for w, s in scores.items() if s.verified}
                    objects = scene_words(vocab)
                    sweep_boards[cfg["verifier"]].append(([(w, scores[w].score) for w in objects], label))
                words = scene_words(vocab, keep)
                metrics = score_board(words, label["expected"], label["not_present"], label.get("ignore"))
                all_words = [w for c in FITZGERALD_ORDER for w in vocab[c]]
                row = {
                    "image": label["image"], "scene": label["scene"], "config": name, "run": run,
                    **metrics,
                    "grounding": grounding_rate(all_words, corpus),
                    "latency_ms": latency_ms + verify_ms,
                    "verify_ms": verify_ms,
                }
                rows.append(row)
                raw.write(json.dumps({**row, "caption": caption, "vocab": vocab, "kept": sorted(keep) if keep is not None else None}) + "\n")
                print(f"    run {run} {name:2}: words={metrics['n_words']} precision={fmt(metrics['precision'])} "
                      f"hard_halluc={fmt(metrics['hard_hallucination'])} recall={fmt(metrics['recall'])}", flush=True)
    raw.close()

    # ---- summary per configuration
    summary = []
    for name in configs:
        r = [row for row in rows if row["config"] == name]
        lat = [row["latency_ms"] for row in r]
        summary.append({
            "config": name, "description": CONFIGS[name]["label"], "boards": len(r),
            "words_per_board": mean_of(r, "n_words"),
            "precision": mean_of(r, "precision"),
            "hard_hallucination": mean_of(r, "hard_hallucination"),
            "not_in_expected": mean_of(r, "not_in_expected"),
            "unlisted": mean_of(r, "unlisted"),
            "recall": mean_of(r, "recall"),
            "grounding": mean_of(r, "grounding"),
            "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95),
        })

    # ---- threshold sweep for the verifiers
    sweep = []
    for verifier_name, boards in sweep_boards.items():
        for threshold in SWEEP[verifier_name]:
            per_board = []
            for scored, label in boards:
                kept = [w for w, s in scored if s is None or s >= threshold]
                per_board.append(score_board(kept, label["expected"], label["not_present"], label.get("ignore")))
            sweep.append({
                "verifier": verifier_name, "threshold": threshold,
                "words_per_board": mean_of(per_board, "n_words"),
                "precision": mean_of(per_board, "precision"),
                "hard_hallucination": mean_of(per_board, "hard_hallucination"),
                "recall": mean_of(per_board, "recall"),
            })

    for filename, data in (("rows.csv", rows), ("summary.csv", summary), ("sweep.csv", sweep)):
        if data:
            with (out_dir / filename).open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)

    # ---- markdown
    reviewed = sum(1 for label in dataset if label.get("reviewed"))
    command = "python eval/run_eval.py " + " ".join(sys.argv[1:])
    cfg = load_config()
    lines = [
        f"## Evaluation run {started:%Y-%m-%d %H:%M}",
        "",
        f"- Command: `{command}`",
        f"- Git commit: `{git_commit()}` · images: {len(dataset)} ({reviewed} with reviewed labels) · runs per config: {args.runs} · temperature 0",
        f"- Gemini models that answered: {dict(model_log.models) or 'unknown'}",
        f"- Verifier thresholds in use: caption_overlap {cfg['thresholds']['caption_overlap']}, blip_itm {cfg['thresholds']['blip_itm']}",
        f"- Raw output: `{out_dir.relative_to(ROOT).as_posix()}/` (raw.jsonl, rows.csv, summary.csv, sweep.csv)",
        "",
        "| Config | Description | Words/board | Precision | Hard hallucination | Not in expected | Unlisted | Recall | AAC grounding | Latency p50 | Latency p95 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summary:
        lines.append(
            f"| {s['config']} | {s['description']} | {fmt(s['words_per_board'], pct=False)} | {fmt(s['precision'])} | "
            f"{fmt(s['hard_hallucination'])} | {fmt(s['not_in_expected'])} | {fmt(s['unlisted'])} | {fmt(s['recall'])} | "
            f"{fmt(s['grounding'])} | {s['latency_p50_ms'] / 1000:.1f}s | {s['latency_p95_ms'] / 1000:.1f}s |"
        )
    if sweep:
        lines += ["", "Verifier threshold sweep (scene words kept at each threshold; 0 = keep all):", "",
                  "| Verifier | Threshold | Words/board | Precision | Hard hallucination | Recall |", "|---|---|---|---|---|---|"]
        for s in sweep:
            lines.append(f"| {s['verifier']} | {s['threshold']} | {fmt(s['words_per_board'], pct=False)} | "
                         f"{fmt(s['precision'])} | {fmt(s['hard_hallucination'])} | {fmt(s['recall'])} |")
    lines += [
        "",
        "Notes: metrics use the board's object words only (protected core words excluded). *Precision* = share in "
        "`expected`; *hard hallucination* = share in `not_present`; *not in expected* = 1 − precision (an upper bound, "
        "as labels don't list every visible object); *unlisted* = in neither list. Latency = caption + RAG (measured "
        "once per image) + Gemini call + verifier; ARASAAC symbol lookup is excluded. C/D score B's boards.",
        "",
    ]
    block = "\n".join(lines)
    print("\n" + block)

    if not RESULTS_MD.exists():
        RESULTS_MD.write_text("# SceneSpeak results\n\nEvery number here comes from a script run recorded below "
                              "(command, date, raw output).\n\n", encoding="utf-8")
    with RESULTS_MD.open("a", encoding="utf-8") as f:
        f.write("\n" + block)
    (out_dir / "summary.md").write_text(block, encoding="utf-8")


if __name__ == "__main__":
    main()
