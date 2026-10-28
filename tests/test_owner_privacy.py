"""Each browser's boards are private to its owner key (app/owner.py)."""

from conftest import OWNER_A, OWNER_B


def make_board(client, photo_bytes, owner, child="Alex"):
    return client.post(
        "/generate-board",
        files={"photo": ("p.png", photo_bytes, "image/png")},
        data={"child_id": child},
        headers={"X-Owner-Key": owner},
    )


def test_board_endpoints_require_an_owner_key(client, photo_bytes):
    assert client.get("/boards").status_code == 401
    assert client.get("/boards/abc").status_code == 401
    resp = client.post("/generate-board", files={"photo": ("p.png", photo_bytes, "image/png")})
    assert resp.status_code == 401


def test_malformed_owner_key_is_rejected(client):
    assert client.get("/boards", headers={"X-Owner-Key": "short"}).status_code == 401
    assert client.get("/boards", headers={"X-Owner-Key": "x" * 40 + "!"}).status_code == 401


def test_same_child_name_is_private_per_owner(client, photo_bytes):
    resp = make_board(client, photo_bytes, OWNER_A)
    assert resp.status_code == 200
    board_id = resp.json()["id"]

    mine = client.get("/boards", params={"child_id": "Alex"}, headers={"X-Owner-Key": OWNER_A}).json()
    theirs = client.get("/boards", params={"child_id": "Alex"}, headers={"X-Owner-Key": OWNER_B}).json()
    assert [b["id"] for b in mine] == [board_id]
    assert theirs == []


def test_another_owners_board_looks_missing(client, photo_bytes):
    board_id = make_board(client, photo_bytes, OWNER_A).json()["id"]
    assert client.get(f"/boards/{board_id}", headers={"X-Owner-Key": OWNER_A}).status_code == 200
    assert client.get(f"/boards/{board_id}", headers={"X-Owner-Key": OWNER_B}).status_code == 404


def test_word_history_is_scoped_to_owner(client, photo_bytes):
    from app.history import get_child_history
    from app.owner import owner_scope, scoped_child

    make_board(client, photo_bytes, OWNER_A)
    a_history = get_child_history(scoped_child(owner_scope(OWNER_A), "Alex"))
    b_history = get_child_history(scoped_child(owner_scope(OWNER_B), "Alex"))
    assert "fridge" in a_history
    assert b_history == []


def test_raw_owner_key_is_never_stored(client, photo_bytes, isolated_storage):
    make_board(client, photo_bytes, OWNER_A)
    db_bytes = (isolated_storage / "test.db").read_bytes()
    assert OWNER_A.encode() not in db_bytes
