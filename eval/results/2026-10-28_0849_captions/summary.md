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
