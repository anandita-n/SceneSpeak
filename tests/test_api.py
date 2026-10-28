"""Basic API behaviour that doesn't need models or the network."""

from conftest import OWNER_A


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_config_without_server_key(client):
    assert client.get("/config").json() == {"server_key": False, "daily_limit": None, "remaining_today": None}


def test_tts_rejects_empty_and_overlong_text(client):
    assert client.get("/tts", params={"text": "  "}).status_code == 400
    assert client.get("/tts", params={"text": "x" * 101}).status_code == 400


def test_non_image_upload_is_rejected(client):
    resp = client.post("/generate-board", files={"photo": ("a.txt", b"hello", "text/plain")},
                       headers={"X-Owner-Key": OWNER_A})
    assert resp.status_code == 400


def test_generated_board_shape(client, photo_bytes):
    resp = client.post("/generate-board", files={"photo": ("p.png", photo_bytes, "image/png")},
                       data={"child_id": "Sam"}, headers={"X-Owner-Key": OWNER_A})
    board = resp.json()
    assert resp.status_code == 200
    assert [e["word"] for e in board["objects"]] == ["fridge", "oven"]
    assert board["verification"]["verifier"] == "none"
    assert all(e["low_confidence"] is False for e in board["objects"])
