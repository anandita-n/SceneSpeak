"""Evaluation metrics, checked against hand-computed examples."""

import pytest

from eval.metrics import grounding_rate, matches, mean_of, percentile, score_board


def test_score_board_hand_example():
    # 4 words: fridge (= refrigerator, expected), swings (= swing, expected),
    # fork (not present), rug (in neither list).
    m = score_board(["fridge", "swings", "fork", "rug"],
                    expected=["refrigerator", "swing", "oven"], not_present=["fork", "bed"])
    assert m["n_words"] == 4
    assert m["precision"] == 2 / 4
    assert m["hard_hallucination"] == 1 / 4
    assert m["not_in_expected"] == 2 / 4
    assert m["unlisted"] == 1 / 4
    assert m["recall"] == pytest.approx(2 / 3)          # oven not covered


def test_ignore_list_is_neither_rewarded_nor_unlisted():
    m = score_board(["cup", "sink"], expected=["sink"], not_present=["bed"], ignore=["cup"])
    assert m["precision"] == 1 / 2 and m["unlisted"] == 0.0


def test_duplicates_and_blanks_are_counted_once():
    m = score_board(["ball", "ball", " "], expected=["ball"], not_present=[])
    assert m["n_words"] == 1 and m["precision"] == 1.0


def test_empty_board():
    m = score_board([], expected=["ball"], not_present=[])
    assert m["n_words"] == 0 and m["precision"] is None and m["recall"] == 0.0


@pytest.mark.parametrize("word,label,expected", [
    ("kids", "person", True),
    ("TV", "television", True),
    ("Swings", "swing", True),
    ("ball", "bowl", False),
    ("ice cream", "ice cream", True),
])
def test_matching(word, label, expected):
    assert matches(word, [label]) is expected


def test_grounding_rate():
    assert grounding_rate(["swing", "Slides", "quantum"], {"swing", "slide"}) == pytest.approx(2 / 3)
    assert grounding_rate([], {"swing"}) is None


def test_percentile_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 50) == 5
    assert percentile(values, 95) == 10
    assert percentile([], 50) is None


def test_mean_of_skips_undefined():
    assert mean_of([{"p": 0.5}, {"p": None}, {"p": 1.0}], "p") == 0.75
    assert mean_of([{"p": None}], "p") is None
