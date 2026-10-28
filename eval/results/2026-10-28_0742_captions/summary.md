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
