# Deploying SceneSpeak

## Hugging Face Spaces (free, Docker)

The free CPU Space (2 vCPU, 16 GB RAM) runs the default setup comfortably. Free Spaces sleep after about
48 hours without visitors and their disk is reset on restart, so saved boards don't survive a restart
unless you add persistent storage (paid).

1. Create a Space at huggingface.co → **New Space** → SDK **Docker** → blank template.
2. Add this block to the very top of the Space's `README.md` (Spaces reads it to know how to run the app):

   ```yaml
   ---
   title: SceneSpeak
   emoji: 🗣️
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. In the Space → **Settings → Variables and secrets**, add:
   - secret `GEMINI_API_KEY` — preferably a **paid-tier** key (on the free tier Google may use submitted
     photos to improve its products)
   - variable `DAILY_BOARD_LIMIT` — e.g. `20`
   - variable `VERIFIER` — `none` (default). `caption_overlap` adds no memory; `blip_itm` downloads a
     ~900 MB model at first use.
4. Push the code to the Space's git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<you>/scenespeak
   git push space main
   ```

5. The first build takes several minutes (CPU PyTorch, BLIP weights and the RAG index are baked into the
   image). Then open the Space URL and make a board.

## Docker on your own machine

```bash
docker compose up --build        # http://127.0.0.1:8000
```

Boards, photos, word history and caches live in the `scenespeak-data` volume. A new volume starts as a
copy of the image's `data/` folder; after you change the AAC word sets, remove the volume
(`docker compose down -v`) to pick up the new file.

## Before real families use it

- Use a paid Gemini key (see above).
- Keep the ARASAAC credit visible wherever symbols appear (already on every board).
- Decide how long photos are kept, and say so on the Create page.
