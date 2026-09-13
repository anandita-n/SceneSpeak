"""Builds a small, labeled test set for evaluating the verification layer
(app/verification.py). These are simple synthetic scenes (not photos) so
the ground truth — which words are actually present — is unambiguous and
reproducible, which is what a precision number needs to mean anything.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "verification_testset"
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def make_mountain_scene() -> str:
    img = Image.new("RGB", (384, 384), "#87CEEB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 280, 384, 384], fill="#228B22")  # grass
    d.polygon([(60, 280), (140, 150), (220, 280)], fill="#8B4513")  # mountain
    d.ellipse([250, 30, 330, 110], fill="#FFD700")  # sun
    path = IMG_DIR / "mountain_scene.png"
    img.save(path)
    return path.name


def make_beach_scene() -> str:
    img = Image.new("RGB", (384, 384), "#87CEEB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 220, 384, 300], fill="#1E90FF")  # water
    d.rectangle([0, 300, 384, 384], fill="#EDC9AF")  # sand
    d.ellipse([280, 20, 350, 90], fill="#FFA500")  # sun
    d.polygon([(100, 300), (150, 220), (200, 300)], fill="#FF6347")  # beach umbrella (triangle)
    d.line([150, 220, 150, 180], fill="#333333", width=4)  # umbrella pole
    path = IMG_DIR / "beach_scene.png"
    img.save(path)
    return path.name


def make_playground_scene() -> str:
    img = Image.new("RGB", (384, 384), "#87CEEB")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 300, 384, 384], fill="#7CFC00")  # grass
    # swing set: two posts + top bar + two swing seats
    d.line([80, 300, 80, 180], fill="#555555", width=6)
    d.line([180, 300, 180, 180], fill="#555555", width=6)
    d.line([80, 180, 180, 180], fill="#555555", width=6)
    d.line([100, 180, 100, 260], fill="#333333", width=3)
    d.rectangle([90, 260, 110, 270], fill="#8B0000")  # seat
    d.line([160, 180, 160, 260], fill="#333333", width=3)
    d.rectangle([150, 260, 170, 270], fill="#8B0000")  # seat
    # slide
    d.polygon([(260, 300), (320, 300), (260, 190)], fill="#FF8C00")
    path = IMG_DIR / "playground_scene.png"
    img.save(path)
    return path.name


LABELS = [
    # (image, word, expected_present)
    ("mountain_scene.png", "mountain", True),
    ("mountain_scene.png", "sun", True),
    ("mountain_scene.png", "grass", True),
    ("mountain_scene.png", "sky", True),
    ("mountain_scene.png", "swing", False),
    ("mountain_scene.png", "car", False),
    ("mountain_scene.png", "book", False),

    ("beach_scene.png", "water", True),
    ("beach_scene.png", "sand", True),
    ("beach_scene.png", "umbrella", True),
    ("beach_scene.png", "sun", True),
    ("beach_scene.png", "mountain", False),
    ("beach_scene.png", "dog", False),
    ("beach_scene.png", "bicycle", False),

    ("playground_scene.png", "swing", True),
    ("playground_scene.png", "slide", True),
    ("playground_scene.png", "grass", True),
    ("playground_scene.png", "sky", True),
    ("playground_scene.png", "water", False),
    ("playground_scene.png", "pizza", False),
    ("playground_scene.png", "computer", False),
]


def main():
    make_mountain_scene()
    make_beach_scene()
    make_playground_scene()
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"image": img, "word": word, "expected_present": present} for img, word, present in LABELS],
            f,
            indent=2,
        )
    print(f"Wrote {len(LABELS)} labeled pairs across 3 images to {OUT_DIR}")


if __name__ == "__main__":
    main()
