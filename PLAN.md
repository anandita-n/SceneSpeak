# Build plan

1. **Core generation loop** ✅ (this commit) — photo → BLIP caption → LLM-generated vocabulary (`/generate-board`).
2. **RAG grounding** — curate a real AAC core-vocabulary + scenario-vocabulary corpus, embed it into ChromaDB, retrieve relevant entries (plus a per-child history table) and feed them into the vocabulary prompt instead of relying on the caption alone.
3. **Symbol mapping** — map each generated word to a real symbol from an open AAC symbol set (e.g. ARASAAC) instead of showing plain text.
4. **Verification layer** — cross-check each generated word against the BLIP caption and/or a CLIP similarity score against the photo; filter out anything that doesn't match. Build a small labeled test set (photo/word pairs, some correct, some deliberately wrong) and report a real precision number.
5. **Speech + persistence** — TTS on tap (Coqui/gTTS); persist boards/vocabulary per child (SQLite for now).
6. **Polish + deploy** — simple frontend, Docker, deploy somewhere demoable.

Build in this order — there's a working, demoable thing after step 1.
