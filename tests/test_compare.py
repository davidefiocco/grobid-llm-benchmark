"""Verify the report parser keys rows by the full heading trail.

GROBID reuses identical matching-mode subheadings under both the Header and Citation
sections, so a parser that keys only on the deepest heading would let the header
``title`` row and citation ``title`` row overwrite each other.
"""

import pytest

from grobid_llm_benchmark.compare import build_comparison, parse_report

pytestmark = pytest.mark.offline

_REPORT = """## Header metadata

#### Strict Matching (exact matches)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| title | 95.0 | 95.0 | 95.0 | 20 |

## Citation metadata

#### Strict Matching (exact matches)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| title | 77.7 | 76.5 | 77.1 | 636 |
| inTitle | 68.7 | 83.0 | 75.2 | 523 |
"""


def test_header_and_citation_title_do_not_collide(tmp_path):
    p = tmp_path / "report.md"
    p.write_text(_REPORT)
    metrics = parse_report(p)
    header_keys = [k for k in metrics if k[0].startswith("Header") and k[1] == "title"]
    citation_keys = [k for k in metrics if k[0].startswith("Citation") and k[1] == "title"]
    assert len(header_keys) == 1
    assert len(citation_keys) == 1
    assert metrics[header_keys[0]].f1 == 95.0
    assert metrics[citation_keys[0]].f1 == 77.1
    # citation-only field present
    assert any(k[1] == "inTitle" for k in metrics)


_GROBID = """## Header metadata

#### Strict Matching (exact matches)

| label | precision | recall | f1 | support |
|---|---|---|---|---|
| title | 95.0 | 95.0 | 95.0 | 20 |
"""

_LLM_A = _GROBID.replace(
    "| title | 95.0 | 95.0 | 95.0 | 20 |", "| title | 80.0 | 80.0 | 80.0 | 20 |"
)
_LLM_B = _GROBID.replace(
    "| title | 95.0 | 95.0 | 95.0 | 20 |", "| title | 90.0 | 90.0 | 90.0 | 20 |"
)


def test_multi_llm_comparison_has_one_column_pair_per_backend(tmp_path):
    g = tmp_path / "grobid.md"
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    g.write_text(_GROBID)
    a.write_text(_LLM_A)
    b.write_text(_LLM_B)

    md = build_comparison(g, [("azure", a), ("ollama", b)])

    # both backends get an f1 + Δ column, sharing the single GROBID column
    assert "| GROBID f1 | azure f1 | azure Δ | ollama f1 | ollama Δ |" in md
    # per-backend f1 and signed delta vs GROBID (80-95=-15.0, 90-95=-5.0) on the title row
    assert "| 95.0 | 80.0 | -15.0 | 90.0 | -5.0 |" in md


def test_single_llm_is_a_rich_pairwise_table(tmp_path):
    g = tmp_path / "grobid.md"
    a = tmp_path / "a.md"
    g.write_text(_GROBID)
    a.write_text(_LLM_A)

    md = build_comparison(g, [("llm", a)])
    # both GROBID and the LLM shown as peer baselines: f1 AND precision/recall for each
    assert "| GROBID f1 | llm f1 | Δ | GROBID P/R | llm P/R |" in md
    assert "| 95.0 | 80.0 | -15.0 | 95.0/95.0 | 80.0/80.0 |" in md
