"""List images in eval/dataset/images that have no label file yet, and labels
that haven't been reviewed.

Usage: python eval/list_unlabeled.py
"""

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parent / "dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    images = sorted(p for p in (DATASET / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    labels = {p.stem: p for p in (DATASET / "labels").glob("*.json")}

    unlabeled = [p.name for p in images if p.stem not in labels]
    unreviewed = [stem for stem, p in sorted(labels.items())
                  if not json.loads(p.read_text(encoding="utf-8")).get("reviewed")]
    missing_image = [stem for stem in sorted(labels) if not any(p.stem == stem for p in images)]

    print(f"{len(images)} images, {len(labels)} label files")
    print(f"\nNo label file ({len(unlabeled)}):", *unlabeled or ["-"], sep="\n  ")
    print(f"\nLabelled but not yet reviewed ({len(unreviewed)}):", *unreviewed or ["-"], sep="\n  ")
    if missing_image:
        print(f"\nLabel without image ({len(missing_image)}) — run eval/coco_bootstrap.py --fetch-only:",
              *missing_image, sep="\n  ")


if __name__ == "__main__":
    main()
