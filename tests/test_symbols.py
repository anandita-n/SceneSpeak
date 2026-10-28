"""ARASAAC lookups: relevance filtering and the memory + disk cache."""

import pytest
import requests

from app import symbols


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or []

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


@pytest.fixture
def fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(symbols, "DB_PATH", tmp_path / "symbols.db")
    symbols.clear_memory_cache()
    yield
    symbols.clear_memory_cache()


def fake_get(responses: dict, calls: list):
    def get(url, timeout):
        term = url.rsplit("/", 1)[-1]
        calls.append(term)
        result = responses[term]
        if isinstance(result, Exception):
            raise result
        return result
    return get


SWING = [{"_id": 4608, "aac": True, "keywords": [{"keyword": "swing"}]}]


def test_found_symbol_is_cached_in_memory_and_on_disk(fresh_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(symbols.requests, "get", fake_get({"swing": FakeResponse(payload=SWING)}, calls))
    entry = symbols.find_symbol("swing")
    assert entry.symbol_id == 4608 and entry.image_url.endswith("4608_500.png")
    symbols.find_symbol("swing")
    assert calls == ["swing"]                      # memory cache
    symbols.clear_memory_cache()
    assert symbols.find_symbol("swing").symbol_id == 4608
    assert calls == ["swing"]                      # disk cache — no second request


def test_irrelevant_results_are_rejected(fresh_cache, monkeypatch):
    garden = [{"_id": 1, "aac": True, "keywords": [{"keyword": "garden"}]}]
    monkeypatch.setattr(symbols.requests, "get", fake_get({"all done": FakeResponse(payload=garden),
                                                           "all": FakeResponse(status_code=404)}, []))
    assert symbols.find_symbol("all done").symbol_id is None


def test_no_symbol_answer_is_cached(fresh_cache, monkeypatch):
    calls = []
    monkeypatch.setattr(symbols.requests, "get", fake_get({"zzz": FakeResponse(status_code=404)}, calls))
    assert symbols.find_symbol("zzz").symbol_id is None
    symbols.clear_memory_cache()
    symbols.find_symbol("zzz")
    assert calls == ["zzz"]


def test_network_failure_is_retried_next_time(fresh_cache, monkeypatch):
    calls = []
    responses = {"ball": requests.ConnectionError("down")}
    monkeypatch.setattr(symbols.requests, "get", fake_get(responses, calls))
    assert symbols.find_symbol("ball").symbol_id is None   # shown as text for now
    responses["ball"] = FakeResponse(payload=[{"_id": 7, "aac": False, "keywords": [{"keyword": "ball"}]}])
    assert symbols.find_symbol("ball").symbol_id == 7      # not stuck on the failure
    assert calls == ["ball", "ball"]


def test_map_words_keeps_order(fresh_cache, monkeypatch):
    monkeypatch.setattr(symbols, "find_symbol", lambda w: symbols.SymbolEntry(word=w))
    assert [e.word for e in symbols.map_words_to_symbols(["c", "a", "b"])] == ["c", "a", "b"]
    assert symbols.map_words_to_symbols([]) == []
