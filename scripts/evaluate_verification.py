"""Evaluates the CLIP-based verification layer against the labeled test
set built by build_test_images.py, and reports precision/recall/accuracy
at a range of thresholds so DEFAULT_THRESHOLD in app/verification.py is a
measured choice, not a guess.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.verification import clip_similarity

TESTSET_DIR = Path(__file__).resolve().parent.parent / "data" / "verification_testset"


def main():
    with open(TESTSET_DIR / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    scored = []
    for entry in labels:
        image = Image.open(TESTSET_DIR / "images" / entry["image"]).convert("RGB")
        score = clip_similarity(image, entry["word"])
        scored.append({**entry, "score": score})
        print(f"{entry['image']:24} {entry['word']:12} expected={entry['expected_present']!s:5} score={score:.4f}")

    print()
    print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'accuracy':>10}")
    best = None
    for threshold in [round(t * 0.01, 2) for t in range(10, 36)]:
        tp = fp = tn = fn = 0
        for s in scored:
            predicted_present = s["score"] >= threshold
            if predicted_present and s["expected_present"]:
                tp += 1
            elif predicted_present and not s["expected_present"]:
                fp += 1
            elif not predicted_present and not s["expected_present"]:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        accuracy = (tp + tn) / len(scored)
        if best is None or accuracy > best[3]:
            best = (threshold, precision, recall, accuracy)
        print(f"{threshold:>10.2f} {precision:>10.2%} {recall:>10.2%} {accuracy:>10.2%}")

    print()
    print(f"Best threshold by accuracy: {best[0]:.2f}  (precision={best[1]:.2%}, recall={best[2]:.2%}, accuracy={best[3]:.2%})")


if __name__ == "__main__":
    main()
