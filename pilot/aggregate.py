"""Aggregate pilot feedback from special educators into RESULTS.md.

Input: a CSV shaped like pilot/responses_template.csv — one row per educator
per board. Reports:

- mean relevance (1-5), overall and per board
- % of words kept = 1 - (words the educator would remove / words on the
  board), averaged over ratings — the same measure QuickPic AAC (Yu et al.,
  2024) reported (speech therapists kept 58% of GPT-3.5's words)
- how many educators would use the tool

Usage: python pilot/aggregate.py pilot/responses.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("educator") and "example row" not in r.get("comments", "")]
    for r in rows:
        r["relevance"] = int(r["relevance"])
        r["words_on_board"] = int(r["words_on_board"])
        r["removed"] = [w.strip() for w in r.get("words_removed", "").split(";") if w.strip()]
        r["added"] = [w.strip() for w in r.get("words_added", "").split(";") if w.strip()]
        if not 1 <= r["relevance"] <= 5:
            raise ValueError(f"relevance must be 1-5: {r}")
    return rows


def summarise(rows: list[dict]) -> dict:
    kept = [1 - len(r["removed"]) / r["words_on_board"] for r in rows if r["words_on_board"]]
    per_board: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        per_board[r["board"]].append(r["relevance"])
    educators = {r["educator"]: r.get("would_use", "").strip().lower() for r in rows}
    return {
        "educators": len(educators),
        "ratings": len(rows),
        "relevance_mean": statistics.mean(r["relevance"] for r in rows),
        "relevance_sd": statistics.stdev(r["relevance"] for r in rows) if len(rows) > 1 else 0.0,
        "kept_mean": statistics.mean(kept),
        "added_per_board": statistics.mean(len(r["added"]) for r in rows),
        "per_board": {b: statistics.mean(v) for b, v in sorted(per_board.items())},
        "would_use": {k: sum(1 for v in educators.values() if v == k) for k in ("yes", "maybe", "no")},
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    rows = load(path)
    if not rows:
        sys.exit("No responses found (the template's example row is ignored).")
    s = summarise(rows)
    lines = [
        f"## Pilot feedback — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Command: `python pilot/aggregate.py {path.as_posix()}`",
        f"- {s['educators']} educators · {s['ratings']} board ratings",
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Mean relevance (1–5) | {s['relevance_mean']:.2f} (SD {s['relevance_sd']:.2f}) |",
        f"| Words educators would keep | {s['kept_mean'] * 100:.1f}% (QuickPic AAC, GPT-3.5: 58%) |",
        f"| Words educators would add, per board | {s['added_per_board']:.1f} |",
        f"| Would use a tool like this | yes {s['would_use']['yes']} · maybe {s['would_use']['maybe']} · no {s['would_use']['no']} |",
        "",
        "Per-board mean relevance: " + ", ".join(f"board {b}: {v:.2f}" for b, v in s["per_board"].items()),
        "",
    ]
    block = "\n".join(lines)
    print(block)
    with (ROOT / "RESULTS.md").open("a", encoding="utf-8") as f:
        f.write("\n" + block)


if __name__ == "__main__":
    main()
