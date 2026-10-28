# SceneSpeak evaluation dataset

About 50 scene photos, each with a label file saying which objects a board
*should* include and which it *must not*. `eval/run_eval.py` scores boards
against these labels.

```
eval/dataset/
  images/         photos (not committed — licences vary; see "Getting the images")
  labels/         one JSON per image (committed)
```

## Label format — `labels/<image stem>.json`

```json
{
  "image": "park_01.jpg",
  "scene": "park",
  "expected": ["swing", "slide", "ball", "tree", "bench"],
  "not_present": ["oven", "bed", "toilet"],
  "ignore": ["cup"],
  "source": "own photo",
  "license": "own work",
  "url": null,
  "reviewed": true
}
```

| Field | Meaning |
|---|---|
| `expected` | Objects **clearly visible** in the photo that a child might want a word for. Use simple nouns ("swing", not "playground swing set"). |
| `not_present` | Objects **not in the photo** that a model might plausibly guess for this kind of scene (e.g. "fork" for a kitchen with no fork). These count as definite mistakes. |
| `ignore` | Objects that are present but tiny or ambiguous — neither rewarded nor penalised. |
| `source` / `license` / `url` | Where the image came from and its licence. Required for any image you didn't take yourself. |
| `reviewed` | `false` for labels pre-filled by `eval/coco_bootstrap.py`; set to `true` once you've checked them. |

Plurals and common synonyms are matched automatically ("swings" = "swing",
"fridge" = "refrigerator", "kid" = "person"), so you don't need to list them.

## What to collect

Aim for **10 scene types × 5 images = 50**:

| Scene type | Examples |
|---|---|
| kitchen | counter, fridge, oven |
| living_room | sofa, TV |
| bedroom | bed, toys |
| bathroom | sink, toilet, bath |
| mealtime | table with food, plates, cups |
| park | benches, trees, playground |
| ball_games | children playing with a ball, frisbee, kite |
| zoo_animals | giraffe, elephant, zebra |
| pets | cat or dog at home |
| transport | bus, train, car |

Mix your own photos with openly licensed ones. Good open sources:
COCO val2017 (labels can be pre-filled — see below), Wikimedia Commons,
Openverse (filter to CC0 / CC BY). **Avoid photos where a child's face is
identifiable**, and never use photos of a real child without consent.

## Getting the images

- **COCO** — `python eval/coco_bootstrap.py --annotations <path to instances_val2017.json>`
  picks 5 images per scene type, downloads them into `images/`, and writes
  pre-filled labels (`reviewed: false`). `expected` comes from COCO's object
  annotations (objects covering ≥ 1% of the image); tiny ones go to `ignore`;
  `not_present` lists the COCO objects that most often appear with this
  image's objects elsewhere in COCO but are absent here — the "adversarial"
  sampling used by POPE (Li et al., EMNLP 2023). Review each file and add
  child-relevant objects COCO doesn't annotate (e.g. "swing", "rug").
- **Already-labelled COCO images** can be re-downloaded on another machine
  with `python eval/coco_bootstrap.py --fetch-only`.
- **Your own photos** — put them in `images/`, run
  `python eval/list_unlabeled.py` to see which still need a label file,
  and write those by hand.
