"""Pydantic models for the structured extraction an LLM must return.

The shape is intentionally close to what the GROBID evaluation scores (header metadata +
a list of bibliographic references), so ``tei_writer`` can map it 1:1 into GROBID-schema
TEI. Author names are split into forename/surname because the evaluation matches on
surnames (``persName/surname``); if a backend only produces a full name string we split
on the last whitespace token as a best effort.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Author(BaseModel):
    forename: str = ""
    surname: str = ""

    @classmethod
    def from_full_name(cls, name: str) -> "Author":
        parts = name.strip().split()
        if not parts:
            return cls()
        if len(parts) == 1:
            return cls(surname=parts[0])
        return cls(forename=" ".join(parts[:-1]), surname=parts[-1])


class Reference(BaseModel):
    """A single bibliographic reference from the article's reference list."""

    title: str = ""  # article/chapter title -> analytic/title
    authors: list[Author] = Field(default_factory=list)
    date: str = ""  # publication year -> monogr/imprint/date/@when
    in_title: str = ""  # journal/book title -> monogr/title
    volume: str = ""
    issue: str = ""
    first_page: str = ""
    doi: str = ""
    pmid: str = ""
    pmcid: str = ""


class Header(BaseModel):
    """Article-level header metadata."""

    title: str = ""
    authors: list[Author] = Field(default_factory=list)
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)


class Body(BaseModel):
    """Full-text structures GROBID's end-to-end evaluation scores (the "Fulltext structures"
    section). The evaluation matches each field **as a bag of text values** (its ``grobidPath``
    collects every matching node and compares the set to the gold set), so these are flat lists
    -- no offsets or nesting needed -- mapped 1:1 into the TEI by ``tei_writer``:

    * ``section_titles``   -> ``//text/body/div/head``           (gold ``//body//sec/title``)
    * ``figure_titles``    -> ``//figure[not(@type)]/head``      (gold ``//fig/label``: the *label*,
      e.g. "Figure 1", not the caption -- GROBID likewise puts the label in ``head``)
    * ``table_titles``     -> ``//figure[@type="table"]/head``   (gold ``//table-wrap/label``: the label)
    * ``citation_markers`` -> ``//ref[@type="bibr"]``            (gold ``//xref[@ref-type="bibr"]``)
    * ``figure_markers``   -> ``//ref[@type="figure"]``          (gold ``//xref[@ref-type="fig"]``)
    * ``table_markers``    -> ``//ref[@type="table"]``           (gold ``//xref[@ref-type="table"]``)
    """

    section_titles: list[str] = Field(default_factory=list)
    figure_titles: list[str] = Field(default_factory=list)
    table_titles: list[str] = Field(default_factory=list)
    citation_markers: list[str] = Field(default_factory=list)
    figure_markers: list[str] = Field(default_factory=list)
    table_markers: list[str] = Field(default_factory=list)


class Extraction(BaseModel):
    """Full LLM extraction result for one article."""

    header: Header = Field(default_factory=Header)
    references: list[Reference] = Field(default_factory=list)
    body: Body = Field(default_factory=Body)


# One-line task description shared by every backend's prompt (kept in sync with the schema).
EXTRACTION_TASK = (
    "Extract, from this scholarly article, the header metadata, the full list of "
    "bibliographic references, and the full-text structures (section titles, figure and "
    "table labels, and the in-text call-out markers for citations, figures and tables)."
)

# JSON schema description embedded in the prompt so the LLM returns the right shape.
EXTRACTION_JSON_HINT = """Return ONLY a JSON object with this exact shape (no prose, no markdown fences):
{
  "header": {
    "title": "string, the article title",
    "authors": [{"forename": "given names", "surname": "family name"}],
    "abstract": "string, the full abstract text",
    "keywords": ["string", ...]
  },
  "references": [
    {
      "title": "title of the cited article/chapter",
      "authors": [{"forename": "given names or initials", "surname": "family name"}],
      "date": "publication year, e.g. 2011",
      "in_title": "journal or book title",
      "volume": "string",
      "issue": "string",
      "first_page": "first page number",
      "doi": "string if present else empty",
      "pmid": "string if present else empty",
      "pmcid": "string if present else empty"
    }
  ],
    "body": {
    "section_titles": ["each section/subsection heading as printed, in reading order, e.g. 'Introduction', 'Materials and Methods'"],
    "figure_titles": ["the LABEL of each figure only (not the caption text), exactly as printed, e.g. 'Figure 1', 'Fig. 2'"],
    "table_titles": ["the LABEL of each table only (not the caption text), exactly as printed, e.g. 'Table 1', 'Table 2'"],
    "citation_markers": ["every in-text citation call-out exactly as printed, one entry per occurrence incl. repeats, e.g. '[12]', '12', '(Smith et al., 2011)'"],
    "figure_markers": ["every in-text figure call-out exactly as printed, one per occurrence, e.g. 'Figure 2', 'Fig. 2'"],
    "table_markers": ["every in-text table call-out exactly as printed, one per occurrence, e.g. 'Table 1'"]
  }
}
Use empty strings or empty arrays for anything not present. Include every reference in the bibliography, and cover the whole document for the body structures (not just the first pages)."""
