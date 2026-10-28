"""The Verifier interface, protected core words, and the verifiers' logic."""

import pytest
from PIL import Image

from app import verification
from app.verification import (
    CaptionOverlapVerifier,
    NoVerifier,
    WordScore,
    get_verifier,
    protected_words,
    verify_board,
)

IMAGE = Image.new("RGB", (8, 8))


class RecordingVerifier:
    """Gives every word the same score and records what it was asked about."""

    name = "recording"

    def __init__(self, score: float, threshold: float = 0.5) -> None:
        self.score = score
        self.threshold = threshold
        self.seen: list[str] = []

    def verify(self, image, words, caption=None):
        self.seen.extend(words)
        return [WordScore(word=w, score=self.score, verified=self.score >= self.threshold) for w in words]


def test_no_verifier_passes_everything_unscored():
    scores = NoVerifier().verify(IMAGE, ["ball", "swing"])
    assert [(s.word, s.score, s.verified) for s in scores] == [("ball", None, True), ("swing", None, True)]


def test_protected_core_words_are_never_sent_to_the_verifier():
    vocab = {"pronouns": ["I"], "verbs": ["want", "push"], "objects": ["more", "swing"]}
    fake = RecordingVerifier(score=0.0)
    scores, _ = verify_board(IMAGE, vocab, "a playground", fake)
    assert set(fake.seen) == {"push", "swing"}           # "want" and "more" are protected
    assert scores["want"].protected and scores["want"].verified
    assert scores["more"].protected and scores["more"].score is None


def test_unverified_categories_are_not_scored():
    vocab = {"prepositions": ["under"], "descriptors": ["shiny"], "objects": ["ball"]}
    fake = RecordingVerifier(score=0.9)
    scores, _ = verify_board(IMAGE, vocab, None, fake)
    assert fake.seen == ["ball"]
    assert scores["under"].score is None and scores["shiny"].score is None


def test_low_scores_are_flagged_not_dropped():
    vocab = {"objects": ["giraffe", "bed"]}
    scores, latency_ms = verify_board(IMAGE, vocab, None, RecordingVerifier(score=0.1))
    assert set(scores) == {"giraffe", "bed"}             # nothing removed
    assert not scores["giraffe"].verified and scores["giraffe"].score == 0.1
    assert latency_ms >= 0


def test_protected_list_comes_from_config():
    words = protected_words()
    assert {"i", "want", "more", "stop", "help", "yes", "no"} <= words


def test_get_verifier_selection(monkeypatch):
    assert get_verifier("none").name == "none"
    assert get_verifier("caption_overlap").name == "caption_overlap"
    monkeypatch.setenv("VERIFIER", "caption_overlap")
    assert get_verifier().name == "caption_overlap"
    with pytest.raises(ValueError):
        get_verifier("clip")


def test_caption_overlap_matches_lemmas_phrases_and_misses(monkeypatch):
    import app.captioning as captioning

    monkeypatch.setattr(captioning, "generate_caption", lambda image: "unused")
    monkeypatch.setattr(captioning, "generate_sampled_captions",
                        lambda image, n, seed=0: ["two swings in a park", "a girl eating ice cream"])
    verifier = CaptionOverlapVerifier(threshold=0.3, samples=2)
    scores = {s.word: s for s in verifier.verify(IMAGE, ["swing", "ice cream", "oven"], caption="children on swings")}
    assert scores["swing"].score == pytest.approx(2 / 3, abs=1e-3) and scores["swing"].verified
    assert scores["ice cream"].score == pytest.approx(1 / 3, abs=1e-3) and scores["ice cream"].verified
    assert scores["oven"].score == 0 and not scores["oven"].verified


def test_empty_word_list():
    assert CaptionOverlapVerifier(0.3, 2).verify(IMAGE, []) == []
    assert verification.BlipItmVerifier("unused", "{word}", 0.5).verify(IMAGE, []) == []
