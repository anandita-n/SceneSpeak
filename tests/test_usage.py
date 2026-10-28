"""Daily board limit for the server's own Gemini key (app/usage.py)."""

from conftest import OWNER_A


def post_board(client, photo_bytes, **data):
    return client.post("/generate-board", files={"photo": ("p.png", photo_bytes, "image/png")},
                       data=data, headers={"X-Owner-Key": OWNER_A})


def test_limit_blocks_after_n_boards_with_server_key(client, photo_bytes, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "server-key")
    monkeypatch.setenv("DAILY_BOARD_LIMIT", "2")
    assert client.get("/config").json() == {"server_key": True, "daily_limit": 2, "remaining_today": 2}
    assert post_board(client, photo_bytes).status_code == 200
    assert post_board(client, photo_bytes).status_code == 200
    blocked = post_board(client, photo_bytes)
    assert blocked.status_code == 429
    assert "limit of 2 boards" in blocked.json()["detail"]
    assert client.get("/config").json()["remaining_today"] == 0


def test_own_key_bypasses_the_limit(client, photo_bytes, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "server-key")
    monkeypatch.setenv("DAILY_BOARD_LIMIT", "0")
    assert post_board(client, photo_bytes).status_code == 429
    assert post_board(client, photo_bytes, gemini_api_key="my-own-key").status_code == 200


def test_failed_generation_refunds_the_slot(client, photo_bytes, monkeypatch):
    import app.main as main
    from app import usage
    from app.vocab import GeminiQuotaError

    monkeypatch.setenv("GEMINI_API_KEY", "server-key")
    monkeypatch.setenv("DAILY_BOARD_LIMIT", "5")

    def quota_error(*a, **k):
        raise GeminiQuotaError("quota")

    monkeypatch.setattr(main, "generate_vocabulary", quota_error)
    resp = post_board(client, photo_bytes)
    assert resp.status_code == 429
    assert usage.used_today() == 0


def test_invalid_limit_value_falls_back_to_default(monkeypatch):
    from app import usage

    monkeypatch.setenv("DAILY_BOARD_LIMIT", "lots")
    assert usage.daily_limit() == usage.DEFAULT_DAILY_LIMIT
