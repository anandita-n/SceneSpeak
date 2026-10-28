"""Pick COCO val2017 images for the evaluation set and pre-fill their labels.

For each scene type, chooses 5 images whose COCO objects match the scene,
preferring permissively licensed images, downloads them into
eval/dataset/images/, and writes eval/dataset/labels/<stem>.json with:

- expected:    COCO objects covering >= 1% of the image (clearly visible)
- ignore:      COCO objects present but smaller than that
- not_present: absent objects that most often co-occur with this image's
               objects across COCO — POPE's "adversarial" negatives
               (Li et al., EMNLP 2023), i.e. the plausible wrong guesses
- source / license / url, and reviewed: false

Labels are a starting point: COCO only annotates 80 object types, so add
child-relevant objects it misses, then set "reviewed": true.

Usage:
  python eval/coco_bootstrap.py --annotations PATH/instances_val2017.json [--per-scene 5]
  python eval/coco_bootstrap.py --fetch-only     # re-download images for existing labels
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import requests

DATASET = Path(__file__).resolve().parent / "dataset"
IMAGES = DATASET / "images"
LABELS = DATASET / "labels"

MIN_AREA_FRACTION = 0.01  # smaller than this -> "ignore", not "expected"
N_NOT_PRESENT = 6

# Prefer the most permissive licences (COCO licence ids): BY, BY-SA,
# no known copyright, US Government, BY-ND, then the non-commercial ones.
LICENSE_PREFERENCE = [4, 5, 7, 8, 6, 1, 2, 3]

# COCO category name -> the word a child's board would use.
CHILD_WORD = {
    "dining table": "table", "potted plant": "plant", "tv": "TV", "cell phone": "phone",
    "sports ball": "ball", "wine glass": "glass", "baseball glove": "glove",
    "baseball bat": "bat", "tennis racket": "racket", "hair drier": "hair dryer",
    "remote": "remote control", "mouse": "computer mouse", "handbag": "bag",
}

FOOD = {"pizza", "sandwich", "cake", "donut", "bowl", "cup", "fork", "spoon", "knife", "banana",
        "apple", "orange", "broccoli", "carrot", "hot dog", "bottle", "wine glass"}

# scene -> list of (category set, minimum number present). All must hold.
SCENES: dict[str, list[tuple[set[str], int]]] = {
    "kitchen": [({"oven", "refrigerator", "microwave", "sink", "toaster"}, 2)],
    "living_room": [({"couch", "tv"}, 2)],
    "bedroom": [({"bed"}, 1)],
    "bathroom": [({"toilet", "sink"}, 2)],
    "mealtime": [({"dining table"}, 1), (FOOD, 2)],
    "park": [({"bench"}, 1), ({"person", "dog", "bird", "umbrella", "bicycle"}, 1)],
    "ball_games": [({"sports ball", "frisbee", "kite"}, 1)],
    "zoo_animals": [({"giraffe", "elephant", "zebra"}, 1)],
    "pets": [({"cat", "dog"}, 1), ({"couch", "bed", "chair", "tv", "laptop", "book"}, 1)],
    "transport": [({"bus", "train"}, 1)],
}


def child_word(category: str) -> str:
    return CHILD_WORD.get(category, category)


def load_coco(path: Path):
    """Return per-image metadata, per-image {category: largest area fraction},
    and co-occurrence counts across the whole val set."""
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    licenses = {lic["id"]: lic for lic in data["licenses"]}
    images = {img["id"]: img for img in data["images"]}

    fractions: dict[int, dict[str, float]] = defaultdict(dict)
    for ann in data["annotations"]:
        img = images[ann["image_id"]]
        frac = ann["area"] / (img["width"] * img["height"])
        name = names[ann["category_id"]]
        fractions[img["id"]][name] = max(frac, fractions[img["id"]].get(name, 0.0))

    count: Counter[str] = Counter()
    pair: Counter[tuple[str, str]] = Counter()
    for cats in fractions.values():
        count.update(cats)
        pair.update(permutations(cats, 2))
    return images, licenses, fractions, count, pair, sorted(set(names.values()))


def adversarial_negatives(present: set[str], all_cats: list[str], count: Counter, pair: Counter) -> list[str]:
    """Absent categories ranked by how strongly they co-occur with the present
    ones: sum over present p of P(c | p)."""
    scores = {c: sum(pair[(p, c)] / count[p] for p in present if count[p])
              for c in all_cats if c not in present}
    return [c for c, s in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if s > 0][:N_NOT_PRESENT]


def fits(scene: str, cats: set[str]) -> bool:
    return all(len(cats & group) >= k for group, k in SCENES[scene])


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    time.sleep(0.2)


def bootstrap(annotations: Path, per_scene: int) -> None:
    images, licenses, fractions, count, pair, all_cats = load_coco(annotations)
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    used: set[int] = set()

    def rank(img_id: int) -> tuple[int, int]:
        lic = images[img_id]["license"]
        return (LICENSE_PREFERENCE.index(lic) if lic in LICENSE_PREFERENCE else 99, img_id)

    for scene in SCENES:
        candidates = []
        for img_id, cats in fractions.items():
            visible = {c for c, f in cats.items() if f >= MIN_AREA_FRACTION}
            if img_id in used or len(visible) < 3 or not fits(scene, set(cats)):
                continue
            candidates.append(img_id)
        chosen = sorted(candidates, key=rank)[:per_scene]
        if len(chosen) < per_scene:
            print(f"warning: only {len(chosen)} images found for {scene}")

        for n, img_id in enumerate(chosen, start=1):
            used.add(img_id)
            img = images[img_id]
            cats = fractions[img_id]
            filename = f"{scene}_{n:02d}_coco{img_id}.jpg"
            download(img["coco_url"], IMAGES / filename)
            lic = licenses[img["license"]]
            label = {
                "image": filename,
                "scene": scene,
                "expected": sorted({child_word(c) for c, f in cats.items() if f >= MIN_AREA_FRACTION}),
                "not_present": [child_word(c) for c in adversarial_negatives(set(cats), all_cats, count, pair)],
                "ignore": sorted({child_word(c) for c, f in cats.items() if f < MIN_AREA_FRACTION}),
                "source": f"COCO val2017 image {img_id}",
                "license": lic["name"],
                "license_url": lic["url"],
                "url": img["flickr_url"],
                "coco_url": img["coco_url"],
                "reviewed": False,
            }
            (LABELS / f"{Path(filename).stem}.json").write_text(json.dumps(label, indent=2) + "\n", encoding="utf-8")
            print(f"{filename}: expected={label['expected']} not_present={label['not_present']} licence={lic['name']}")


def fetch_only() -> None:
    """Download any COCO images referenced by labels but missing locally."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    for path in sorted(LABELS.glob("*.json")):
        label = json.loads(path.read_text(encoding="utf-8"))
        if label.get("coco_url") and not (IMAGES / label["image"]).exists():
            download(label["coco_url"], IMAGES / label["image"])
            print("downloaded", label["image"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, help="path to COCO instances_val2017.json")
    parser.add_argument("--per-scene", type=int, default=5)
    parser.add_argument("--fetch-only", action="store_true")
    args = parser.parse_args()
    if args.fetch_only:
        fetch_only()
    elif args.annotations:
        bootstrap(args.annotations, args.per_scene)
    else:
        parser.error("give --annotations PATH, or --fetch-only")


if __name__ == "__main__":
    main()
