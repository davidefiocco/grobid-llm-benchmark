"""Produce GROBID TEI for each article PDF via a running GROBID REST service.

Calls ``POST /api/processFulltextDocument`` on the configured GROBID server and writes the
result next to each PDF as ``<name>.fulltext.tei.xml`` -- the exact suffix the evaluator
reads for the GROBID baseline (``jatsEval -Prun=0``). Decouples GROBID processing (the
service, which owns the models) from scoring (which only reads TEI + gold).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from grobid_llm_benchmark.dataset import find_article_dirs
from grobid_llm_benchmark.pdf import find_pdf

GROBID_TEI_SUFFIX = ".fulltext.tei.xml"


@dataclass
class GrobidRunSummary:
    server: str
    n_articles: int
    n_ok: int
    n_failed: int
    total_seconds: float
    failures: list[str] = field(default_factory=list)


def is_alive(server: str, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(f"{server.rstrip('/')}/api/isalive", timeout=timeout)
        return r.status_code == 200 and "true" in r.text.lower()
    except httpx.HTTPError:
        return False


def process_pdf(
    server: str,
    pdf_path: Path,
    timeout: float,
    consolidate_citations: int,
    consolidate_header: int = 1,
) -> str:
    """Return the GROBID TEI for one PDF via processFulltextDocument.

    Header and citation consolidation are independent knobs, mirroring GROBID's own end-to-end
    evaluator (``consolidateHeader=1``, ``consolidateCitations=0``): that is the flow behind the
    published PMC/bioRxiv/PLOS benchmark numbers. Citation consolidation is off by default because
    it *replaces* parsed reference fields with full-precision CrossRef metadata (e.g. a full
    ``2010-04-21`` date vs the year-only JATS gold), which the scorer compares verbatim and which
    therefore deflates citation scores rather than lifting them. Both consolidation modes require
    a reachable consolidation service (glutton); for a fully offline baseline pass 0 for both.
    """
    url = f"{server.rstrip('/')}/api/processFulltextDocument"
    with open(pdf_path, "rb") as fh:
        files = {"input": (pdf_path.name, fh, "application/pdf")}
        data = {
            "consolidateHeader": str(consolidate_header),
            "consolidateCitations": str(consolidate_citations),
            "includeRawCitations": "0",
        }
        r = httpx.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text


def run_grobid(
    data_dir: Path,
    server: str,
    timeout: float = 300.0,
    consolidate_citations: int = 0,
    consolidate_header: int = 1,
    overwrite: bool = False,
    limit: int | None = None,
    progress=None,
) -> GrobidRunSummary:
    data_dir = Path(data_dir)
    dirs = find_article_dirs(data_dir)
    if limit is not None:
        dirs = dirs[:limit]

    ok = 0
    failures: list[str] = []
    start = time.time()
    for article_dir in dirs:
        pdf = find_pdf(article_dir)
        if pdf is None:
            continue
        out = pdf.with_name(pdf.stem + GROBID_TEI_SUFFIX)
        if out.exists() and not overwrite:
            ok += 1
            if progress:
                progress(article_dir.name, True)
            continue
        try:
            tei = process_pdf(server, pdf, timeout, consolidate_citations, consolidate_header)
            out.write_text(tei, encoding="utf-8")
            ok += 1
            if progress:
                progress(article_dir.name, True)
        except (httpx.HTTPError, OSError) as e:
            failures.append(f"{article_dir.name}: {type(e).__name__}: {e}")
            if progress:
                progress(article_dir.name, False)

    return GrobidRunSummary(
        server=server,
        n_articles=len(dirs),
        n_ok=ok,
        n_failed=len(failures),
        total_seconds=round(time.time() - start, 2),
        failures=failures,
    )
