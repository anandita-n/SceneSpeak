# SceneSpeak results

Every number here comes from a script run recorded below (command, date, raw output).
Resume claims should only use numbers from this file.

## RAG and child-history consistency — 2026-10-26

- Command: `python scripts/evaluate_rag_consistency.py <image dir> --runs 3 --out report.json`
- Commit: `9ed2004` · 14 images (12 COCO val2017 + 2 own) · 3 runs per condition · model default temperature
- Raw output: `data/evaluation/rag_consistency.json`

```
MEAN over 14 images:
  without_rag            consistency=0.731 corpus_alignment=0.591
  with_rag               consistency=0.724 corpus_alignment=0.649
  with_rag_and_history   consistency=0.960 corpus_alignment=0.658
```

Consistency = mean pairwise Jaccard overlap of a board's words across repeated runs of the
same photo. Corpus alignment = share of words found in the curated AAC word sets.

## Evaluation run 2026-10-28 07:09

- Command: `python eval/run_eval.py --limit 5 --runs 3`
- Git commit: `9ca910e` · images: 5 (0 with reviewed labels) · runs per config: 3 · temperature 0
- Gemini models that answered: {'gemini-flash-lite-latest': 45}
- Verifier thresholds in use: caption_overlap 0.3, blip_itm 0.5
- Raw output: `eval/results/2026-10-28_0709/` (raw.jsonl, rows.csv, summary.csv, sweep.csv)

| Config | Description | Words/board | Precision | Hard hallucination | Not in expected | Unlisted | Recall | AAC grounding | Latency p50 | Latency p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | Caption only (no photo), no RAG | 5.00 | 41.9% | 0.0% | 58.1% | 58.1% | 58.0% | 60.3% | 5.3s | 7.7s |
| A | LLM with photo, no RAG | 7.80 | 42.1% | 1.6% | 57.9% | 44.3% | 82.6% | 64.5% | 6.2s | 8.7s |
| B | LLM + RAG | 7.53 | 34.8% | 1.7% | 65.2% | 52.5% | 75.0% | 74.8% | 5.7s | 9.1s |
| C | LLM + RAG + caption_overlap | 2.07 | 80.1% | 0.0% | 19.9% | 19.9% | 46.0% | 74.8% | 10.2s | 11.2s |
| D | LLM + RAG + blip_itm | 0.00 | – | – | – | – | 0.0% | 74.8% | 7.5s | 14.4s |

Verifier threshold sweep (scene words kept at each threshold; 0 = keep all):

| Verifier | Threshold | Words/board | Precision | Hard hallucination | Recall |
|---|---|---|---|---|---|
| caption_overlap | 0.0 | 7.53 | 34.8% | 1.7% | 75.0% |
| caption_overlap | 0.33 | 2.07 | 80.1% | 0.0% | 46.0% |
| caption_overlap | 0.66 | 1.07 | 90.0% | 0.0% | 30.7% |
| caption_overlap | 1.0 | 0.87 | 90.0% | 0.0% | 24.0% |
| blip_itm | 0.0 | 7.53 | 34.8% | 1.7% | 75.0% |
| blip_itm | 0.01 | 6.07 | 33.2% | 1.9% | 63.3% |
| blip_itm | 0.02 | 4.87 | 44.3% | 2.9% | 59.3% |
| blip_itm | 0.05 | 3.00 | 59.0% | 0.0% | 47.7% |
| blip_itm | 0.1 | 0.67 | 66.7% | 0.0% | 11.7% |
| blip_itm | 0.2 | 0.00 | – | – | 0.0% |
| blip_itm | 0.3 | 0.00 | – | – | 0.0% |
| blip_itm | 0.5 | 0.00 | – | – | 0.0% |

Notes: metrics use the board's object words only (protected core words excluded). *Precision* = share in `expected`; *hard hallucination* = share in `not_present`; *not in expected* = 1 − precision (an upper bound, as labels don't list every visible object); *unlisted* = in neither list. Latency = caption + RAG (measured once per image) + Gemini call + verifier; ARASAAC symbol lookup is excluded. C/D score B's boards.

## Captioning evaluation 2026-10-28 07:42

- Command: `python eval/caption_eval.py --captions C:/Users/adi/.cache/scenespeak/coco/captions_val2017.json`
- Git commit: `afaa12c` · images: 50 COCO val2017 eval images · references: COCO's 5 human captions each
- Fine-tuned model: `C:\Users\adi\GenAi\checkpoints\blip-finetuned` (GenAi project; config: 400 Flickr8k training images, 3 epochs, CPU)
- Gemini models that answered: {'gemini-flash-lite-latest': 50}
- Raw output: `eval/results/2026-10-28_0742_captions/` (captions.jsonl, summary.csv)

| Captioner | BLEU-4 | METEOR | ROUGE-L | CIDEr | Words/caption | Median time |
|---|---|---|---|---|---|---|
| blip_base | 0.4053 | 0.5214 | 0.5606 | 1.2884 | 8.9 | 6.74s |
| blip_base_scenespeak | 0.4155 | 0.5160 | 0.5601 | 1.2769 | 8.9 | 3.80s |
| blip_finetuned | 0.4180 | 0.5426 | 0.5669 | 1.3401 | 10.2 | 6.36s |
| gemini | 0.2521 | 0.5193 | 0.5141 | 1.0161 | 11.5 | 2.23s |

Example captions (first 3 images):

- `ball_games_01_coco100238.jpg` — reference: “A family standing next to each other while holding frisbees.”
  - blip_base: “a group of people”
  - blip_base_scenespeak: “a woman holding a frth”
  - blip_finetuned: “a group of people standing next to each other people”
  - gemini: “Two men, a woman, and a child holding frisbees.”
- `ball_games_02_coco139872.jpg` — reference: “A dog laying on grass near a flying disc.”
  - blip_base: “a black and white dog laying in the grass”
  - blip_base_scenespeak: “a black and white dog laying in the grass”
  - blip_finetuned: “a black and white dog laying in the grass with a red frisbee”
  - gemini: “A happy black-and-white border collie rests beside a red frisbee on green grass.”
- `ball_games_03_coco239773.jpg` — reference: “Baseball player in blue and white uniform swinging at a ball.”
  - blip_base: “a baseball player holding a bat on a field”
  - blip_base_scenespeak: “a baseball player swinging at a pitch during a game”
  - blip_finetuned: “a man in a blue shirt and white pants holding a baseball bat”
  - gemini: “A baseball player in a blue jersey holding a bat.”

Metrics mirror the GenAi project's `src/evaluate.py` (lower-cased, punctuation stripped). Gemini was prompted for a one-sentence caption at temperature 0; its time is the API round trip (BLIP times are local CPU inference).

## Latency profile — baseline — 2026-10-28 08:49

- Command: `python scripts/profile_board.py --label baseline --images 5`
- Git commit: `0271659` (plus uncommitted changes if any) · 5 boards · fresh server process, warmed up · verifier: config default

| Stage | Median | Max |
|---|---|---|
| caption | 3.73s | 4.31s |
| retrieval | 0.03s | 0.04s |
| llm | 3.15s | 4.42s |
| symbols_and_verify | 1.57s | 4.42s |
| save | 0.01s | 0.01s |
| **total (server)** | 8.49s | 11.65s |

- End-to-end wall time: p50 8.50s, p95 11.71s
- Worst `/health` response while a board was generating: median 8297 ms, max 11468 ms (how long other requests were stuck waiting)
- Raw per-board timings: `[{"caption": 2765, "retrieval": 35, "llm": 4425, "symbols_and_verify": 4416, "save": 7, "total": 11649}, {"caption": 3726, "retrieval": 43, "llm": 1681, "symbols_and_verify": 2343, "save": 7, "total": 7799}, {"caption": 4306, "retrieval": 33, "llm": 2209, "symbols_and_verify": 1188, "save": 7, "total": 7742}, {"caption": 3723, "retrieval": 33, "llm": 3149, "symbols_and_verify": 1575, "save": 10, "total": 8490}, {"caption": 4020, "retrieval": 31, "llm": 3431, "symbols_and_verify": 1151, "save": 8, "total": 8642}]`
- Test boards removed afterwards: 5

## Captioning evaluation 2026-10-28 08:49

- Command: `python eval/caption_eval.py --captions C:/Users/adi/.cache/scenespeak/coco/captions_val2017.json --models blip_base_scenespeak,blip_base_norep,blip_base_greedy`
- Git commit: `0271659` · images: 50 COCO val2017 eval images · references: COCO's 5 human captions each
- Fine-tuned model: not run
- Gemini: not run
- Raw output: `eval/results/2026-10-28_0849_captions/` (captions.jsonl, summary.csv)

| Captioner | BLEU-4 | METEOR | ROUGE-L | CIDEr | Words/caption | Median time |
|---|---|---|---|---|---|---|
| blip_base_scenespeak | 0.4155 | 0.5160 | 0.5601 | 1.2769 | 8.9 | 3.61s |
| blip_base_norep | 0.4127 | 0.5159 | 0.5624 | 1.2882 | 8.9 | 3.59s |
| blip_base_greedy | 0.3751 | 0.4820 | 0.5675 | 1.2106 | 7.6 | 1.51s |

Example captions (first 3 images):

- `ball_games_01_coco100238.jpg` — reference: “A family standing next to each other while holding frisbees.”
  - blip_base_scenespeak: “a woman holding a frth”
  - blip_base_norep: “a woman holding a frth”
  - blip_base_greedy: “a man and woman holding frth frths”
- `ball_games_02_coco139872.jpg` — reference: “A dog laying on grass near a flying disc.”
  - blip_base_scenespeak: “a black and white dog laying in the grass”
  - blip_base_norep: “a black and white dog laying in the grass”
  - blip_base_greedy: “a dog laying on the grass”
- `ball_games_03_coco239773.jpg` — reference: “Baseball player in blue and white uniform swinging at a ball.”
  - blip_base_scenespeak: “a baseball player swinging at a pitch during a game”
  - blip_base_norep: “a baseball player swinging a bat”
  - blip_base_greedy: “a baseball player”

Metrics mirror the GenAi project's `src/evaluate.py` (lower-cased, punctuation stripped). Gemini was prompted for a one-sentence caption at temperature 0; its time is the API round trip (BLIP times are local CPU inference).

## Latency profile — after (empty symbol cache) — 2026-10-28 08:59

- Command: `python scripts/profile_board.py --label after (empty symbol cache) --images 5`
- Git commit: `0271659` (plus uncommitted changes if any) · 5 boards · fresh server process, warmed up · verifier: config default

| Stage | Median | Max |
|---|---|---|
| caption | 3.42s | 3.94s |
| retrieval | 0.03s | 0.03s |
| llm | 1.75s | 2.98s |
| symbols_and_verify | 2.22s | 4.09s |
| save | 0.01s | 0.01s |
| **total (server)** | 7.73s | 8.54s |

- End-to-end wall time: p50 7.76s, p95 8.60s
- Worst `/health` response while a board was generating: median 28 ms, max 35 ms (how long other requests were stuck waiting)
- Raw per-board timings: `[{"caption": 2676, "retrieval": 25, "llm": 1750, "symbols_and_verify": 4087, "save": 6, "total": 8544}, {"caption": 3533, "retrieval": 30, "llm": 1708, "symbols_and_verify": 2452, "save": 6, "total": 7730}, {"caption": 3941, "retrieval": 28, "llm": 1944, "symbols_and_verify": 2222, "save": 6, "total": 8141}, {"caption": 3421, "retrieval": 28, "llm": 1711, "symbols_and_verify": 1438, "save": 7, "total": 6604}, {"caption": 3345, "retrieval": 28, "llm": 2981, "symbols_and_verify": 1260, "save": 6, "total": 7620}]`
- Test boards removed afterwards: 5

## Latency profile — after (symbol cache on disk) — 2026-10-28 09:00

- Command: `python scripts/profile_board.py --label after (symbol cache on disk) --images 5`
- Git commit: `0271659` (plus uncommitted changes if any) · 5 boards · fresh server process, warmed up · verifier: config default

| Stage | Median | Max |
|---|---|---|
| caption | 3.40s | 3.93s |
| retrieval | 0.03s | 0.03s |
| llm | 2.79s | 3.87s |
| symbols_and_verify | 1.15s | 1.23s |
| save | 0.01s | 0.01s |
| **total (server)** | 6.47s | 8.46s |

- End-to-end wall time: p50 6.48s, p95 8.47s
- Worst `/health` response while a board was generating: median 27 ms, max 41 ms (how long other requests were stuck waiting)
- Raw per-board timings: `[{"caption": 2657, "retrieval": 30, "llm": 1624, "symbols_and_verify": 1226, "save": 7, "total": 5543}, {"caption": 3402, "retrieval": 29, "llm": 3868, "symbols_and_verify": 1155, "save": 6, "total": 8461}, {"caption": 3927, "retrieval": 28, "llm": 2789, "symbols_and_verify": 1153, "save": 7, "total": 7904}, {"caption": 3552, "retrieval": 30, "llm": 2063, "symbols_and_verify": 813, "save": 6, "total": 6464}, {"caption": 3359, "retrieval": 29, "llm": 3055, "symbols_and_verify": 21, "save": 5, "total": 6469}]`
- Test boards removed afterwards: 5
- Note on the two "after" profiles above: both include the Phase 3 changes (sync board endpoint, ARASAAC disk cache, no repetition penalty). The second run's server also started after the prompt's RAG wording change (Phase 4, `rag_mode: wording`) was saved, so its LLM stage isn't strictly comparable. Gemini latency varies ~1.6–3.9 s per call, so total-time differences across 5 boards are within noise; the event-loop fix (other requests stuck 8,297 ms → 28 ms median) is the robust result.
