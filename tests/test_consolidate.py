"""Consolidation mapping with a stubbed glutton client (no network).

Verifies the CrossRef->Reference/Header mapping and the mode semantics (replace vs
identifiers-only vs off), plus graceful handling of a no-match.
"""

import pytest

from grobid_llm_benchmark.consolidate import (
    _crossref_date,
    consolidate_extraction,
)
from grobid_llm_benchmark.models import Author, Extraction, Header, Reference

pytestmark = pytest.mark.offline

_CROSSREF_RECORD = {
    "DOI": "10.1000/xyz123",
    "title": ["Canonical Reference Title"],
    "container-title": ["Journal of Canonical Studies"],
    "author": [
        {"given": "Jane", "family": "Doe", "sequence": "first"},
        {"given": "John", "family": "Roe", "sequence": "additional"},
    ],
    "volume": "42",
    "issue": "3",
    "page": "100-120",
    "issued": {"date-parts": [[2011, 5, 3]]},
    "pmid": 12345,
    "pmcid": "PMC67890",
}


class _StubGlutton:
    """Returns the canned record for any DOI or title/author lookup; None otherwise."""

    def __init__(self, record):
        self.record = record
        self.calls = []

    def lookup(self, **params):
        self.calls.append(params)
        if params.get("doi") or params.get("atitle"):
            return self.record
        return None


def _extraction():
    return Extraction(
        header=Header(title="Some Paper", authors=[Author(surname="Smith")]),
        references=[
            Reference(title="rough parsed title", authors=[Author(surname="Doe")], date="2011")
        ],
    )


def test_crossref_date_builds_full_precision():
    assert _crossref_date({"issued": {"date-parts": [[2011, 5, 3]]}}) == "2011-05-03"
    assert _crossref_date({"issued": {"date-parts": [[1999]]}}) == "1999"
    assert _crossref_date({}) == ""


def test_citations_mode_1_replaces_fields():
    ext = _extraction()
    client = _StubGlutton(_CROSSREF_RECORD)

    summary = consolidate_extraction(ext, client, citations_mode=1, header_mode=0)

    ref = ext.references[0]
    assert summary.references_matched == 1
    assert ref.title == "Canonical Reference Title"
    assert ref.in_title == "Journal of Canonical Studies"
    assert [a.surname for a in ref.authors] == ["Doe", "Roe"]
    assert ref.volume == "42"
    assert ref.first_page == "100"
    assert ref.date == "2011-05-03"
    assert ref.doi == "10.1000/xyz123"
    assert ref.pmid == "12345"
    assert ref.pmcid == "PMC67890"


def test_citations_mode_2_keeps_fields_sets_ids_only():
    ext = _extraction()
    client = _StubGlutton(_CROSSREF_RECORD)

    consolidate_extraction(ext, client, citations_mode=2, header_mode=0)

    ref = ext.references[0]
    assert ref.title == "rough parsed title"  # unchanged
    assert ref.date == "2011"  # unchanged
    assert ref.doi == "10.1000/xyz123"  # identifier filled
    assert ref.pmid == "12345"


def test_header_mode_1_replaces_title_and_authors():
    ext = _extraction()
    client = _StubGlutton(_CROSSREF_RECORD)

    summary = consolidate_extraction(ext, client, citations_mode=0, header_mode=1)

    assert summary.header_matched is True
    assert ext.header.title == "Canonical Reference Title"
    assert [a.surname for a in ext.header.authors] == ["Doe", "Roe"]


def test_no_match_leaves_extraction_untouched():
    ext = _extraction()
    ext.references[0].title = ""  # nothing to match on
    ext.references[0].authors = []
    client = _StubGlutton(None)

    summary = consolidate_extraction(ext, client, citations_mode=1, header_mode=0)

    assert summary.references_matched == 0
    assert ext.references[0].title == ""
