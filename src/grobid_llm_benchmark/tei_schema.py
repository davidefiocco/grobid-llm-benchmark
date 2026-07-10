"""The exact GROBID TEI structure that ``EndToEndEvaluation`` scores.

GROBID's Java end-to-end evaluation
(``grobid-trainer/.../evaluation/EndToEndEvaluation.java`` +
``.../evaluation/utilities/FieldSpecification.java``) compares two XML documents per
article using XPath:

* the **gold** JATS/NLM file (``.nxml``), via each field's ``nlmPath``;
* the **tool output** TEI file (``*.fulltext.tei.xml`` for GROBID), via each field's
  ``grobidPath``.

To make an LLM directly comparable, we emit a TEI document with **the same element
structure GROBID produces**, so the identical ``grobidPath`` XPaths select the LLM's
values. This module records those XPaths (the contract our ``tei_writer`` must honour)
and nothing else -- it is the single source of truth for what actually gets scored.

Namespace: TEI ``http://www.tei-c.org/ns/1.0`` (prefix ``tei`` in the evaluator).
Matching modes applied by the evaluator to textual fields: strict, soft (case/space/
punct-insensitive), Levenshtein (>=0.8), Ratcliff/Obershelp (>=0.95). Numeric/date
fields (date, volume, issue, page, ids) use strict matching only.
"""

from __future__ import annotations

# --- Header fields actually added to the evaluation (others are commented out in
# FieldSpecification.setUpFields) and the GROBID TEI XPath used to select them. -------
HEADER_FIELDS_GROBID_XPATH: dict[str, str] = {
    "title": "//titleStmt/title/text()",
    "authors": "//sourceDesc/biblStruct/analytic/author/persName/surname/text()",
    "first_author": "//sourceDesc/biblStruct/analytic/author[1]/persName/surname/text()",
    "abstract": "//profileDesc/abstract//text()",
    "keywords": "//profileDesc/textClass/keywords//text()",
}

# --- Citation base path + per-citation fields (relative to each biblStruct). ----------
CITATION_BASE_GROBID_XPATH = "//back/div/listBibl/biblStruct"
CITATION_FIELDS_GROBID_XPATH: dict[str, str] = {
    "title": "analytic/title/text()",
    "authors": "analytic/author/persName/surname/text()",
    "first_author": "analytic/author[1]/persName/surname/text()",
    "date": "monogr/imprint/date/@when",
    "inTitle": "monogr/title/text()",
    "volume": 'monogr/imprint/biblScope[@unit="volume" or @unit="vol"]/text()',
    "issue": 'monogr/imprint/biblScope[@unit="issue"]/text()',
    "page": 'monogr/imprint/biblScope[@unit="page"]/@from',
    "id": "@id",
    "doi": 'analytic/idno[@type="DOI"]/text()',
    "pmid": 'analytic/idno[@type="PMID"]/text()',
    "pmcid": 'analytic/idno[@type="PMCID"]/text()',
}

# --- Fulltext structures (the evaluation's "Fulltext structures" section). Matched
# field-level as a bag of text values, so tei_writer only needs the right nodes to exist
# (no offsets/order). Keys mirror FieldSpecification's fulltextLabels. -----------------
FULLTEXT_FIELDS_GROBID_XPATH: dict[str, str] = {
    "section_title": "//text/body/div/head/text()",
    "figure_title": "//figure[not(@type)]/head/text()",
    "table_title": '//figure[@type="table"]/head/text()',
    "reference_citation": '//ref[@type="bibr"]/text()',
    "reference_figure": '//ref[@type="figure"]/text()',
    "reference_table": '//ref[@type="table"]/text()',
}

# Reference/citation-context linkage ids (used for instance-level citation mapping):
GROBID_CITATION_CONTEXT_ID = '//ref[@type="bibr"]/@target'
GROBID_BIB_REFERENCE_ID = "//listBibl/biblStruct/@id"

TEI_NS = "http://www.tei-c.org/ns/1.0"

# Suffix under which the harness writes LLM-produced TEI next to each PDF. The patched
# grobid-trainer LLM run type reads this suffix (mirroring GROBID's ".fulltext.tei.xml"),
# and it can be overridden at score time via -Pllmsuffix to score a specific backend.
LLM_TEI_SUFFIX = ".fulltext.llm.tei.xml"


def llm_tei_suffix(tag: str | None = None) -> str:
    """TEI suffix for an LLM run, optionally namespaced by a backend ``tag``.

    With no tag this is the canonical ``.fulltext.llm.tei.xml`` (what the scorer reads by
    default). A tag inserts a segment -- e.g. ``azure`` -> ``.fulltext.llm.azure.tei.xml``
    -- so several backends' TEI can coexist next to the same PDF and each be scored
    independently (pass the same suffix to the scorer via ``-Pllmsuffix``) without
    re-running any other backend or GROBID.
    """
    tag = (tag or "").strip().strip(".")
    if not tag:
        return LLM_TEI_SUFFIX
    return f".fulltext.llm.{tag}.tei.xml"
