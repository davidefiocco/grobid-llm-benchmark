"""Render an :class:`Extraction` into GROBID-schema TEI.

The output mirrors the element structure GROBID emits in ``*.fulltext.tei.xml`` so that
the *same* XPaths in ``FieldSpecification`` (see :mod:`tei_schema`) select the LLM's
values during scoring. Only the elements the evaluation actually reads are populated:

header   -> teiHeader/fileDesc/{titleStmt/title, sourceDesc/biblStruct/analytic/author},
            teiHeader/profileDesc/{abstract, textClass/keywords}
refs     -> text/back/div[@type="references"]/listBibl/biblStruct  (matches
            ``//back/div/listBibl/biblStruct``)
fulltext -> text/body: one ``div/head`` per section title; ``figure/head`` (figures) and
            ``figure[@type="table"]/head`` (tables); and ``ref[@type="bibr|figure|table"]``
            call-out markers. These feed the evaluation's "Fulltext structures" fields,
            which match field-level as bags of text values (no offsets/order needed).
"""

from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

from grobid_llm_benchmark.models import Author, Body, Extraction, Reference
from grobid_llm_benchmark.tei_schema import LLM_TEI_SUFFIX, TEI_NS

_NSMAP = {None: TEI_NS}

# XML 1.0 forbids most control characters. Strip everything outside the legal ranges so
# noisy PDF text or LLM output cannot produce an unserialisable / unparseable TEI file.
_ILLEGAL_XML = re.compile("[^\u0009\u000a\u000d\u0020-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def _clean(text: str) -> str:
    return _ILLEGAL_XML.sub("", text)


# The gold @from holds a single first-page number; models sometimes return a range
# ("123-145") or "pp. 123". Keep only the leading page token so strict page matching works.
_FIRST_PAGE = re.compile(r"\d+[A-Za-z]?|[A-Za-z]?\d+")


def _first_page(value: str) -> str:
    m = _FIRST_PAGE.search(value)
    return m.group(0) if m else value.strip()


# Gold figure/table structures carry only the *label* ("Figure 1", "Table 2"), but a VLM
# often returns the whole caption. When the string is longer than a bare label, reduce it
# to the leading "Fig(ure)? N" / "Table N" token; otherwise keep it as-is.
_LABEL = re.compile(r"^\s*((?:fig(?:ure|s?\.?)?|table|tab\.?|scheme|plate)\s*[0-9IVXLC]+)", re.I)


def _label_only(caption: str) -> str:
    m = _LABEL.match(caption)
    return m.group(1).strip() if m else caption.strip()


def _el(parent, tag, text=None, **attrs):
    e = etree.SubElement(parent, f"{{{TEI_NS}}}{tag}", nsmap=None)
    for k, v in attrs.items():
        e.set(k.replace("__", ":"), _clean(v))
    if text is not None and text != "":
        e.text = _clean(text)
    return e


def _pers_name(author_parent, author: Author):
    """Emit <author><persName>...<surname>..</surname></persName></author>."""
    author_el = _el(author_parent, "author")
    pers = _el(author_el, "persName")
    if author.forename:
        _el(pers, "forename", author.forename, type="first")
    if author.surname:
        _el(pers, "surname", author.surname)
    return author_el


def _add_reference(list_bibl, idx: int, ref: Reference):
    bibl = _el(list_bibl, "biblStruct")
    # GROBID emits xml:id on biblStruct, but its evaluator resolves the citation id via the
    # unprefixed relative XPath "@id" (FieldSpecification citationIdField / grobidBibReferenceId).
    # Whether "@id" matches "xml:id" depends on the XPath engine's namespace handling, so we set
    # both: xml:id mirrors GROBID's output, plain id guarantees the evaluator's "@id" resolves.
    bibl.set("{http://www.w3.org/XML/1998/namespace}id", f"b{idx}")
    bibl.set("id", f"b{idx}")

    analytic = _el(bibl, "analytic")
    if ref.title:
        _el(analytic, "title", ref.title, level="a", type="main")
    for a in ref.authors:
        if a.surname or a.forename:
            _pers_name(analytic, a)
    for idno_type, value in (("DOI", ref.doi), ("PMID", ref.pmid), ("PMCID", ref.pmcid)):
        if value:
            _el(analytic, "idno", value, type=idno_type)

    monogr = _el(bibl, "monogr")
    if ref.in_title:
        _el(monogr, "title", ref.in_title, level="j")
    imprint = _el(monogr, "imprint")
    if ref.volume:
        _el(imprint, "biblScope", ref.volume, unit="volume")
    if ref.issue:
        _el(imprint, "biblScope", ref.issue, unit="issue")
    if ref.first_page:
        _el(imprint, "biblScope", None, unit="page", **{"from": _first_page(ref.first_page)})
    if ref.date:
        _el(imprint, "date", ref.date, type="published", when=ref.date)


def _build_body(body_el, body: Body) -> None:
    """Populate ``text/body`` with the fulltext structures the evaluation scores.

    Section titles become ``div/head`` (one div each), figure/table labels become
    ``figure/head`` (tables carry ``@type="table"``), and in-text call-outs become
    ``ref[@type="bibr|figure|table"]`` markers inside a single paragraph. All are matched
    field-level as text bags, so their grouping/order is irrelevant to scoring.
    """
    for title in body.section_titles:
        if title and title.strip():
            div = _el(body_el, "div")
            _el(div, "head", title)

    for caption in body.figure_titles:
        if caption and caption.strip():
            fig = _el(body_el, "figure")
            _el(fig, "head", _label_only(caption))

    for caption in body.table_titles:
        if caption and caption.strip():
            fig = _el(body_el, "figure", type="table")
            _el(fig, "head", _label_only(caption))

    # Call-out markers carry no @target: the LLM extracts marker text but not which
    # biblStruct each resolves to, so citation-context linkage (a separate "Citation context
    # resolution" block, not part of the scored Citation-metadata F1) is out of scope here.
    markers = (
        [("bibr", m) for m in body.citation_markers]
        + [("figure", m) for m in body.figure_markers]
        + [("table", m) for m in body.table_markers]
    )
    if any(m for _, m in markers):
        p = _el(_el(body_el, "div"), "p")
        for ref_type, marker in markers:
            if marker and marker.strip():
                _el(p, "ref", marker, type=ref_type)


def build_tei(extraction: Extraction) -> etree._ElementTree:
    root = etree.Element(f"{{{TEI_NS}}}TEI", nsmap=_NSMAP)
    root.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    tei_header = _el(root, "teiHeader")
    file_desc = _el(tei_header, "fileDesc")

    title_stmt = _el(file_desc, "titleStmt")
    if extraction.header.title:
        _el(title_stmt, "title", extraction.header.title, level="a", type="main")

    source_desc = _el(file_desc, "sourceDesc")
    bibl_struct = _el(source_desc, "biblStruct")
    analytic = _el(bibl_struct, "analytic")
    for a in extraction.header.authors:
        if a.surname or a.forename:
            _pers_name(analytic, a)
    _el(analytic, "title", extraction.header.title or "", level="a", type="main")
    _el(bibl_struct, "monogr")

    profile_desc = _el(tei_header, "profileDesc")
    abstract = _el(profile_desc, "abstract")
    if extraction.header.abstract:
        _el(abstract, "p", extraction.header.abstract)
    if extraction.header.keywords:
        text_class = _el(profile_desc, "textClass")
        keywords = _el(text_class, "keywords")
        for kw in extraction.header.keywords:
            _el(keywords, "term", kw)

    text_el = _el(root, "text")
    body_el = _el(text_el, "body")
    _build_body(body_el, extraction.body)
    back = _el(text_el, "back")
    refs_div = _el(back, "div", type="references")
    list_bibl = _el(refs_div, "listBibl")
    for i, ref in enumerate(extraction.references):
        _add_reference(list_bibl, i, ref)

    return etree.ElementTree(root)


def write_tei(extraction: Extraction, pdf_path: Path, suffix: str = LLM_TEI_SUFFIX) -> Path:
    """Write the TEI next to the PDF as ``<name><suffix>`` and return its path.

    ``suffix`` defaults to the canonical ``.fulltext.llm.tei.xml``; pass a backend-tagged
    suffix (see :func:`tei_schema.llm_tei_suffix`) to keep multiple backends' TEI side by
    side.
    """
    tree = build_tei(extraction)
    out_path = pdf_path.with_name(pdf_path.stem + suffix)
    tree.write(str(out_path), xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return out_path
