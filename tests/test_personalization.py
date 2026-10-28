"""Caregiver edits are saved and change the next board for that child."""

from conftest import OWNER_A, OWNER_B


def board(client, photo_bytes, child="Alex", owner=OWNER_A):
    resp = client.post("/generate-board", files={"photo": ("p.png", photo_bytes, "image/png")},
                       data={"child_id": child}, headers={"X-Owner-Key": owner})
    assert resp.status_code == 200, resp.text
    return resp.json()


def edit(client, board_id, action, word, category="objects", owner=OWNER_A):
    return client.post(f"/boards/{board_id}/edits", headers={"X-Owner-Key": owner},
                       json={"action": action, "word": word, "category": category})


def words(b, category="objects"):
    return [e["word"] for e in b[category]]


def test_edits_are_saved_on_the_board(client, photo_bytes):
    first = board(client, photo_bytes)
    assert words(first) == ["fridge", "oven"]
    assert edit(client, first["id"], "remove", "oven").status_code == 200
    updated = edit(client, first["id"], "add", "toaster").json()
    assert words(updated) == ["fridge", "toaster"]
    reopened = client.get(f"/boards/{first['id']}", headers={"X-Owner-Key": OWNER_A}).json()
    assert words(reopened) == ["fridge", "toaster"]


def test_an_edit_changes_the_next_board(client, photo_bytes, monkeypatch):
    import app.main as main
    from app.vocab import VocabularySet

    calls = []

    def fake_llm(*args, **kwargs):
        calls.append(kwargs)
        # The model keeps suggesting "oven" — the caregiver's removal must win.
        return VocabularySet(pronouns=["I"], verbs=["cook"], prepositions=["in"],
                             descriptors=["hot"], objects=["fridge", "oven"])

    monkeypatch.setattr(main, "generate_vocabulary", fake_llm)

    first = board(client, photo_bytes)
    edit(client, first["id"], "remove", "oven")
    edit(client, first["id"], "add", "toaster")

    second = board(client, photo_bytes)                     # same kind of scene, same child
    assert "oven" not in words(second)                      # removed word excluded
    assert calls[-1]["removed_words"] == ["oven"]           # and the model was told
    assert calls[-1]["added_words"] == ["toaster"]          # added word passed as context
    assert "toaster" in calls[-1]["child_history"]          # and part of the child's wording


def test_edits_only_affect_that_child(client, photo_bytes, monkeypatch):
    import app.main as main
    from app.vocab import VocabularySet

    calls = []
    monkeypatch.setattr(main, "generate_vocabulary", lambda *a, **k: calls.append(k) or VocabularySet(
        objects=["fridge", "oven"]))

    first = board(client, photo_bytes, child="Alex")
    edit(client, first["id"], "remove", "oven")
    other = board(client, photo_bytes, child="Sam")
    assert "oven" in words(other)
    assert calls[-1]["removed_words"] == []


def test_cannot_edit_another_owners_board(client, photo_bytes):
    first = board(client, photo_bytes)
    assert edit(client, first["id"], "remove", "oven", owner=OWNER_B).status_code == 404


def test_edit_validation(client, photo_bytes):
    first = board(client, photo_bytes)
    assert edit(client, first["id"], "rename", "oven").status_code == 422
    assert edit(client, first["id"], "add", "x" * 41).status_code == 422
    assert edit(client, first["id"], "add", "lamp", category="colours").status_code == 422
