"""Profile board generation through a real server process.

Starts the app with uvicorn (no reload), waits for its startup warm-up,
then generates one board per eval image. While each board is generating, a
second thread pings /health every 200 ms — if the server blocks its event
loop, those pings stall, which is what other users would feel.

Reports per-stage latency (from the board's own timings), total p50/p95,
and /health latency during generation; appends the result to RESULTS.md.

Uses the GEMINI_API_KEY from .env as a per-request key, so profiling doesn't
count toward the server's daily board limit. Boards it creates are deleted.

Usage: python scripts/profile_board.py --label baseline [--images 5]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.metrics import percentile  # noqa: E402
from eval.run_eval import load_dataset  # noqa: E402

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
OWNER = "ProfileOwnerKey_0123456789abcdef"
CHILD = "__profile__"


def start_server() -> subprocess.Popen:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )
    warmed = threading.Event()

    def drain() -> None:  # read server output; note when the warm-up finishes
        for line in proc.stdout:
            if "warm-up done" in line:
                warmed.set()

    threading.Thread(target=drain, daemon=True).start()
    if not warmed.wait(timeout=300):
        proc.terminate()
        raise RuntimeError("server warm-up didn't finish within 5 minutes")
    return proc


def ping_health(stop: threading.Event, out: list[float]) -> None:
    while not stop.is_set():
        t = time.perf_counter()
        try:
            requests.get(f"{BASE}/health", timeout=60)
            out.append((time.perf_counter() - t) * 1000)
        except requests.RequestException:
            out.append(60_000.0)
        time.sleep(0.2)


def cleanup() -> int:
    db = sqlite3.connect(ROOT / "data" / "scenespeak.db")
    rows = db.execute("SELECT image_filename FROM boards WHERE child_id LIKE ?", (f"%:{CHILD}",)).fetchall()
    for (filename,) in rows:
        (ROOT / "data" / "uploads" / filename).unlink(missing_ok=True)
    db.execute("DELETE FROM boards WHERE child_id LIKE ?", (f"%:{CHILD}",))
    db.execute("DELETE FROM child_vocabulary WHERE child_id LIKE ?", (f"%:{CHILD}",))
    db.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="e.g. baseline / after")
    parser.add_argument("--images", type=int, default=5)
    args = parser.parse_args()
    key = dotenv_values(ROOT / ".env").get("GEMINI_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY in .env first.")

    images = load_dataset(args.images)
    proc = start_server()
    results = []
    try:
        for label in images:
            path = ROOT / "eval" / "dataset" / "images" / label["image"]
            pings: list[float] = []
            stop = threading.Event()
            pinger = threading.Thread(target=ping_health, args=(stop, pings))
            pinger.start()
            t = time.perf_counter()
            resp = requests.post(
                f"{BASE}/generate-board", headers={"X-Owner-Key": OWNER},
                files={"photo": (path.name, path.read_bytes(), "image/jpeg")},
                data={"child_id": CHILD, "gemini_api_key": key}, timeout=300,
            )
            wall_ms = (time.perf_counter() - t) * 1000
            stop.set()
            pinger.join()
            resp.raise_for_status()
            timings = resp.json()["timings_ms"]
            results.append({"image": label["image"], "wall_ms": wall_ms, "timings": timings,
                            "health_max_ms": max(pings) if pings else None,
                            "health_median_ms": statistics.median(pings) if pings else None})
            print(f"{label['image']}: wall={wall_ms:.0f}ms {timings} health_max={results[-1]['health_max_ms']:.0f}ms", flush=True)
            time.sleep(4)  # free-tier rate limit
    finally:
        proc.terminate()
        removed = cleanup()

    stages = [k for k in results[0]["timings"] if k != "total"]
    walls = [r["wall_ms"] for r in results]
    health = [r["health_max_ms"] for r in results if r["health_max_ms"] is not None]
    lines = [
        f"## Latency profile — {args.label} — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"- Command: `python scripts/profile_board.py --label {args.label} --images {args.images}`",
        f"- Git commit: `{subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}`"
        f" (plus uncommitted changes if any) · {len(results)} boards · fresh server process, warmed up · verifier: "
        f"{os.environ.get('VERIFIER', 'config default')}",
        "",
        "| Stage | Median | Max |",
        "|---|---|---|",
    ]
    for stage in stages:
        vals = [r["timings"][stage] for r in results]
        lines.append(f"| {stage} | {statistics.median(vals) / 1000:.2f}s | {max(vals) / 1000:.2f}s |")
    lines += [
        f"| **total (server)** | {statistics.median([r['timings']['total'] for r in results]) / 1000:.2f}s | "
        f"{max(r['timings']['total'] for r in results) / 1000:.2f}s |",
        "",
        f"- End-to-end wall time: p50 {percentile(walls, 50) / 1000:.2f}s, p95 {percentile(walls, 95) / 1000:.2f}s",
        f"- Worst `/health` response while a board was generating: median {statistics.median(health):.0f} ms, "
        f"max {max(health):.0f} ms (how long other requests were stuck waiting)",
        f"- Raw per-board timings: `{json.dumps([r['timings'] for r in results])}`",
        f"- Test boards removed afterwards: {removed}",
        "",
    ]
    block = "\n".join(lines)
    print("\n" + block)
    with (ROOT / "RESULTS.md").open("a", encoding="utf-8") as f:
        f.write("\n" + block)


if __name__ == "__main__":
    main()
