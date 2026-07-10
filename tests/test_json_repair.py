"""Verify lenient/truncation-tolerant JSON parsing of model responses.

A model can hit the token cap mid-generation, so parsing must recover the completed
portion (a partial references list) rather than discard the whole article.
"""

import pytest

from grobid_llm_benchmark.json_utils import coerce_json as _coerce_json

pytestmark = pytest.mark.offline


def test_plain_json():
    assert _coerce_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert _coerce_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_truncated_inside_string_value():
    # cut off mid-string deep inside a reference (the real failure mode observed)
    text = (
        '{"header": {"title": "T", "authors": [{"forename": "A", "surname": "B"}], '
        '"abstract": "x", "keywords": []}, '
        '"references": [{"title": "R1", "authors": [{"forename": "C", "surname": "D"}], '
        '"date": "2011", "in_title": "British Journal of Clinical Pharmacol'
    )
    out = _coerce_json(text)
    assert set(out) >= {"header", "references"}
    # the one completed reference is preserved
    assert len(out["references"]) == 1
    assert out["references"][0]["title"] == "R1"


def test_truncated_after_comma():
    text = (
        '{"header": {"title": "T", "authors": []}, '
        '"references": [{"title": "R1"}, {"title": "R2"},'
    )
    out = _coerce_json(text)
    assert len(out["references"]) == 2


def test_no_json_raises():
    with pytest.raises(ValueError):
        _coerce_json("this is prose with no object")
