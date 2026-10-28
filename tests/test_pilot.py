"""Pilot feedback aggregation, checked against a hand-computed example."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pilot"))
from aggregate import load, summarise  # noqa: E402


def test_summary_hand_example(tmp_path):
    csv_file = tmp_path / "responses.csv"
    csv_file.write_text(
        "educator,role,board,relevance,words_on_board,words_removed,words_added,would_use,comments\n"
        "E1,SLP,1,4,20,cup;napkin,plate,yes,\n"          # kept 18/20 = 0.90
        "E1,SLP,2,2,10,a;b;c;d;e,,yes,\n"                # kept 5/10  = 0.50
        "E2,teacher,1,5,20,,,maybe,\n"                   # kept 20/20 = 1.00
        "E9,x,1,1,10,,,no,example row - delete before use\n",  # template row: ignored
        encoding="utf-8",
    )
    s = summarise(load(csv_file))
    assert s["educators"] == 2 and s["ratings"] == 3
    assert s["relevance_mean"] == pytest.approx(11 / 3)
    assert s["kept_mean"] == pytest.approx((0.9 + 0.5 + 1.0) / 3)
    assert s["added_per_board"] == pytest.approx(1 / 3)
    assert s["per_board"] == {"1": 4.5, "2": 2.0}
    assert s["would_use"] == {"yes": 1, "maybe": 1, "no": 0}


def test_relevance_out_of_range_is_rejected(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("educator,role,board,relevance,words_on_board,words_removed,words_added,would_use,comments\n"
                        "E1,SLP,1,7,20,,,yes,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load(csv_file)
