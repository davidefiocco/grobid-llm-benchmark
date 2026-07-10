"""Verify the TEI writer emits exactly the structure GROBID's evaluation XPaths read."""

import re

import pytest

from grobid_llm_benchmark import tei_schema as S
from grobid_llm_benchmark.models import Author, Body, Extraction, Header, Reference
from grobid_llm_benchmark.tei_writer import build_tei

pytestmark = pytest.mark.offline

NS = {"tei": S.TEI_NS}


def _teiify(xpath: str) -> str:
    """Prefix unprefixed element steps with ``tei:`` so lxml (namespace-aware) matches.

    GROBID's Java evaluator parses the TEI with a *non* namespace-aware DOM, so its
    unprefixed paths match by local name. lxml is namespace-aware, so for the test we
    add the ``tei:`` prefix to element name steps to reproduce the same selection.
    """
    return re.sub(r"(?<![\w:@\*])([a-zA-Z][\w-]*)(?=[/\[]|\$|/text|\Z)", r"tei:\1", xpath)


def _sample() -> Extraction:
    return Extraction(
        header=Header(
            title="A Great Paper",
            authors=[
                Author(forename="Jane", surname="Doe"),
                Author(forename="John", surname="Smith"),
            ],
            abstract="This is the abstract.",
            keywords=["calcium", "neuron"],
        ),
        references=[
            Reference(
                title="Ref One",
                authors=[Author(forename="A", surname="Alpha")],
                date="1997",
                in_title="Mol. Neurobiol",
                volume="15",
                issue="2",
                first_page="131",
                doi="10.1/x",
            ),
            Reference(
                title="Ref Two", authors=[Author(surname="Beta")], date="2001", in_title="PNAS"
            ),
        ],
        body=Body(
            section_titles=["Introduction", "Methods"],
            figure_titles=["Figure 1"],
            table_titles=["Table 1"],
            citation_markers=["[1]", "[2]"],
            figure_markers=["Figure 1"],
            table_markers=["Table 1"],
        ),
    )


def test_header_fields_resolve():
    root = build_tei(_sample()).getroot()
    got = {
        f: root.xpath(_teiify(xp), namespaces=NS) for f, xp in S.HEADER_FIELDS_GROBID_XPATH.items()
    }
    assert got["title"] == ["A Great Paper"]
    assert got["authors"] == ["Doe", "Smith"]
    assert got["first_author"] == ["Doe"]
    assert got["abstract"] == ["This is the abstract."]
    assert got["keywords"] == ["calcium", "neuron"]


def test_llm_tei_suffix_tagging():
    assert S.llm_tei_suffix() == ".fulltext.llm.tei.xml"
    assert S.llm_tei_suffix("") == ".fulltext.llm.tei.xml"
    assert S.llm_tei_suffix("azure") == ".fulltext.llm.azure.tei.xml"
    # tolerate surrounding dots/whitespace in the tag
    assert S.llm_tei_suffix(" .ollama. ") == ".fulltext.llm.ollama.tei.xml"


def test_fulltext_fields_resolve():
    root = build_tei(_sample()).getroot()
    got = {
        f: root.xpath(_teiify(xp), namespaces=NS)
        for f, xp in S.FULLTEXT_FIELDS_GROBID_XPATH.items()
    }
    assert got["section_title"] == ["Introduction", "Methods"]
    assert got["figure_title"] == ["Figure 1"]
    assert got["table_title"] == ["Table 1"]
    assert got["reference_citation"] == ["[1]", "[2]"]
    assert got["reference_figure"] == ["Figure 1"]
    assert got["reference_table"] == ["Table 1"]


def test_citation_fields_resolve():
    root = build_tei(_sample()).getroot()
    bibls = root.xpath(_teiify(S.CITATION_BASE_GROBID_XPATH), namespaces=NS)
    assert len(bibls) == 2
    first = {
        f: bibls[0].xpath(_teiify(xp), namespaces=NS)
        for f, xp in S.CITATION_FIELDS_GROBID_XPATH.items()
    }
    assert first["title"] == ["Ref One"]
    assert first["authors"] == ["Alpha"]
    assert first["date"] == ["1997"]
    assert first["inTitle"] == ["Mol. Neurobiol"]
    assert first["volume"] == ["15"]
    assert first["page"] == ["131"]
    assert first["doi"] == ["10.1/x"]
    assert first["id"] == ["b0"]
