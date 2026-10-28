# SceneSpeak pilot — feedback form for special educators

Thank you for helping. This takes about **15–20 minutes**. You'll look at 5 communication boards that
SceneSpeak generated from photos of everyday scenes and tell us how useful they'd be for the children
you work with.

**About SceneSpeak.** It's a student research project, not a medical device. It turns a photo of a scene
(a kitchen, a park, bath time) into an AAC board of picture symbols in Fitzgerald Key order. No real
children's photos are used in this pilot — the boards come from public scene photos.

**Your privacy.** Please don't include your name or any child's name. You'll be identified only by the
code we give you (e.g. `E1`). You can stop at any time; we'll only report averages.

---

Your code: `____`  Your role (e.g. special educator, SLP, teaching assistant): `____________`
Years working with AAC users: `____`

For **each board** (you'll get a link or a printout labelled Board 1–5):

### Board `__`

1. **Relevance** — How well do the words fit the scene for a child who uses AAC?
   `1` not useful · `2` a few useful words · `3` somewhat useful · `4` useful with small changes · `5` ready to use
   **Rating:** `__`

2. **Words you would remove** — list any words you'd delete before using this board
   (wrong for the scene, too hard, not useful):
   `_______________________________________________`

3. **Words you would add** — any important words that are missing:
   `_______________________________________________`

4. **Comments** (optional): `_______________________________________________`

(Repeat for Boards 2–5.)

---

### Overall (optional)

- Would you use a tool like this to prepare boards for a new activity? `yes / maybe / no`
- What would make it more useful? `_______________________________________________`

---

## For the researcher: recording responses

Enter each educator's answers into `pilot/responses.csv` (copy `pilot/responses_template.csv`), one row
per educator per board. In `words_on_board`, put the number of words the board had; in `words_removed`,
list the words they'd remove separated by `;`. Then run:

```bash
python pilot/aggregate.py pilot/responses.csv
```

It reports mean relevance and the share of words educators would keep — measured the same way as
QuickPic AAC (Yu et al., 2024), where speech therapists kept 58% of GPT-3.5's words — and appends the
result to `RESULTS.md`.
