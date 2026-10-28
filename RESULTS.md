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
