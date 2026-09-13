# Build plan

1. **Core generation loop** ✅ — photo → BLIP caption → LLM-generated vocabulary (`/generate-board`).
2. **RAG grounding** ✅ — a curated AAC core-vocabulary + scenario-vocabulary corpus, embedded into ChromaDB; the most relevant entries are retrieved per scene and combined with a per-child history table (SQLite) so vocabulary stays consistent across a child's boards over time.
3. **Symbol mapping** ✅ — each generated word is mapped to a real ARASAAC pictogram via their public API, with a relevance-scoring layer on top (ARASAAC's raw search isn't ranked by relevance, so a naive "first result" pick returned wrong symbols for phrases — e.g. "all done" matched "garden" — fixed by only trusting a result when the query and one of its own keywords actually overlap). Words with no good match degrade gracefully to text-only rather than showing a wrong symbol.
4. **Verification layer** — cross-check each generated word against the BLIP caption and/or a CLIP similarity score against the photo; filter out anything that doesn't match. Build a small labeled test set (photo/word pairs, some correct, some deliberately wrong) and report a real precision number.
5. **Speech + persistence** — TTS on tap (Coqui/gTTS); persist boards/vocabulary per child (SQLite for now).
6. **Polish + deploy** — simple frontend, Docker, deploy somewhere demoable.

Build in this order — there's a working, demoable thing after step 1.
