# Build plan

1. **Core generation loop** ✅ — photo → BLIP caption → LLM-generated vocabulary (`/generate-board`).
2. **RAG grounding** ✅ — a curated AAC core-vocabulary + scenario-vocabulary corpus, embedded into ChromaDB; the most relevant entries are retrieved per scene and combined with a per-child history table (SQLite) so vocabulary stays consistent across a child's boards over time.
3. **Symbol mapping** ✅ — each generated word is mapped to a real ARASAAC pictogram via their public API, with a relevance-scoring layer on top (ARASAAC's raw search isn't ranked by relevance, so a naive "first result" pick returned wrong symbols for phrases — e.g. "all done" matched "garden" — fixed by only trusting a result when the query and one of its own keywords actually overlap). Words with no good match degrade gracefully to text-only rather than showing a wrong symbol.
4. **Verification layer** ✅ — each generated "object" word is scored against the real photo with CLIP and filtered out if the similarity falls below a measured threshold. Built a 21-pair labeled test set (`data/verification_testset`, `scripts/build_test_images.py`) with known ground truth and swept thresholds (`scripts/evaluate_verification.py`) to pick one rather than guessing: **90.9% precision, 83.3% recall, 85.7% accuracy** at threshold 0.23. Deliberately scoped to the "objects" category only — core vocabulary and prepositions aren't things a photo depicts, so CLIP similarity isn't a meaningful check for them.
5. **Speech + persistence** — TTS on tap (Coqui/gTTS); persist boards/vocabulary per child (SQLite for now).
6. **Polish + deploy** — simple frontend, Docker, deploy somewhere demoable.

Build in this order — there's a working, demoable thing after step 1.
