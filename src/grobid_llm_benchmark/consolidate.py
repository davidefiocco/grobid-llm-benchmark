"""Optional glutton-backed consolidation for LLM extractions.

Enrich parsed metadata against CrossRef via a running biblio-glutton service, matching GROBID's
two independent knobs -- ``consolidateHeader`` and ``consolidateCitations`` -- each with the same
three modes:

* ``0`` -- off (no lookup).
* ``1`` -- replace: overwrite parsed fields with the matched CrossRef record (canonical title,
  authors, journal, volume/issue/page, date, and identifiers).
* ``2`` -- identifiers only: keep the parsed fields, just fill in DOI/PMID/PMCID when matched.

Consolidation runs on the structured :class:`Extraction` before TEI is written, using the same
lookup order as GROBID (DOI first, then metadata matching). As with GROBID, citation consolidation
replaces reference dates with full-precision CrossRef dates, which mismatch the year-only JATS gold
under the end-to-end scorer, so it is kept off the scored citation comparison (see README >
Consolidation). Header consolidation is safe to score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from grobid_llm_benchmark.models import Author, Extraction, Header, Reference

_YEAR_RE = re.compile(r"\d{4}")


@dataclass
class ConsolidationSummary:
    references_total: int = 0
    references_matched: int = 0
    header_matched: bool = False


class GluttonClient:
    """Thin client over biblio-glutton's ``GET /service/lookup`` endpoint.

    Any transport error, non-200 status, or empty/invalid body is treated as "no match"
    (returns ``None``) so consolidation degrades gracefully to the parsed metadata.
    """

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def __enter__(self) -> "GluttonClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def lookup(
        self,
        *,
        doi: str = "",
        atitle: str = "",
        first_author: str = "",
        jtitle: str = "",
        volume: str = "",
        first_page: str = "",
        year: str = "",
        biblio: str = "",
    ) -> dict | None:
        params: dict[str, str] = {}
        if doi:
            params["doi"] = doi
        if atitle:
            params["atitle"] = atitle
        if first_author:
            params["firstAuthor"] = first_author
        if jtitle:
            params["jtitle"] = jtitle
        if volume:
            params["volume"] = volume
        if first_page:
            params["firstPage"] = first_page
        if year:
            params["year"] = year
        if biblio:
            params["biblio"] = biblio
        if not params:
            return None
        try:
            resp = self._client.get(f"{self.base_url}/service/lookup", params=params)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        if not isinstance(data, dict) or not data:
            return None
        return data


def _first(value) -> str:
    """CrossRef renders title/container-title as arrays; take the first non-empty entry."""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    if value is None:
        return ""
    return str(value).strip()


def _year_of(date: str) -> str:
    m = _YEAR_RE.search(date or "")
    return m.group(0) if m else ""


def _crossref_date(record: dict) -> str:
    """Build a ``YYYY[-MM[-DD]]`` string from the first available CrossRef date-parts."""
    for key in ("issued", "published-print", "published-online", "published", "created"):
        block = record.get(key)
        if not isinstance(block, dict):
            continue
        parts = block.get("date-parts")
        if not (isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]):
            continue
        nums = [p for p in parts[0] if isinstance(p, int)]
        if not nums:
            continue
        out = f"{nums[0]:04d}"
        if len(nums) > 1:
            out += f"-{nums[1]:02d}"
        if len(nums) > 2:
            out += f"-{nums[2]:02d}"
        return out
    return ""


def _crossref_authors(record: dict) -> list[Author]:
    authors: list[Author] = []
    for a in record.get("author") or []:
        if not isinstance(a, dict):
            continue
        family, given = a.get("family", ""), a.get("given", "")
        if family or given:
            authors.append(Author(forename=given or "", surname=family or ""))
    return authors


def _first_author_surname(authors: list[Author]) -> str:
    return authors[0].surname if authors and authors[0].surname else ""


def _apply_identifiers(record: dict, *, set_doi, set_pmid, set_pmcid) -> None:
    doi = record.get("DOI") or record.get("doi") or ""
    pmid = record.get("pmid")
    pmcid = record.get("pmcid") or record.get("pmc")
    if doi:
        set_doi(str(doi))
    if pmid:
        set_pmid(str(pmid))
    if pmcid:
        set_pmcid(str(pmcid))


def _enrich_reference(ref: Reference, record: dict, mode: int) -> None:
    _apply_identifiers(
        record,
        set_doi=lambda v: setattr(ref, "doi", v),
        set_pmid=lambda v: setattr(ref, "pmid", v),
        set_pmcid=lambda v: setattr(ref, "pmcid", v),
    )
    if mode != 1:
        return
    title = _first(record.get("title"))
    if title:
        ref.title = title
    jtitle = _first(record.get("container-title"))
    if jtitle:
        ref.in_title = jtitle
    authors = _crossref_authors(record)
    if authors:
        ref.authors = authors
    if record.get("volume"):
        ref.volume = str(record["volume"])
    if record.get("issue"):
        ref.issue = str(record["issue"])
    page = record.get("page")
    if page:
        ref.first_page = str(page).split("-")[0].strip()
    date = _crossref_date(record)
    if date:
        ref.date = date


def _lookup_reference(ref: Reference, client: GluttonClient) -> dict | None:
    """DOI-first, then metadata matching -- the same order GROBID's consolidation uses."""
    if ref.doi:
        record = client.lookup(doi=ref.doi)
        if record is not None:
            return record
    return client.lookup(
        atitle=ref.title,
        first_author=_first_author_surname(ref.authors),
        jtitle=ref.in_title,
        volume=ref.volume,
        first_page=ref.first_page,
        year=_year_of(ref.date),
    )


def _consolidate_header(header: Header, client: GluttonClient, mode: int) -> bool:
    record = client.lookup(atitle=header.title, first_author=_first_author_surname(header.authors))
    if record is None:
        return False
    if mode == 1:
        title = _first(record.get("title"))
        if title:
            header.title = title
        authors = _crossref_authors(record)
        if authors:
            header.authors = authors
    return True


def consolidate_extraction(
    extraction: Extraction,
    client: GluttonClient,
    *,
    citations_mode: int = 0,
    header_mode: int = 0,
) -> ConsolidationSummary:
    """Enrich ``extraction`` in place via glutton and return a match summary."""
    summary = ConsolidationSummary(references_total=len(extraction.references))
    if citations_mode:
        for ref in extraction.references:
            record = _lookup_reference(ref, client)
            if record is not None:
                _enrich_reference(ref, record, citations_mode)
                summary.references_matched += 1
    if header_mode:
        summary.header_matched = _consolidate_header(extraction.header, client, header_mode)
    return summary
