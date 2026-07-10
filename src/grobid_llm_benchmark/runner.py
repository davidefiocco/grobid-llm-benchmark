"""Orchestrate the LLM pass over a dataset directory.

For each article directory (containing a PDF + gold ``.nxml``) we run the configured LLM
backend on the PDF, render the result into GROBID-schema TEI written as
``*.fulltext.llm.tei.xml`` next to the PDF, and record per-article latency and any
failures. A JSON run summary is written so the comparison step can report throughput and
failure counts alongside GROBID's precision/recall/F1.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from grobid_llm_benchmark.backends.base import BackendConfig, get_backend
from grobid_llm_benchmark.consolidate import GluttonClient, consolidate_extraction
from grobid_llm_benchmark.pdf import find_pdfs
from grobid_llm_benchmark.tei_schema import LLM_TEI_SUFFIX
from grobid_llm_benchmark.tei_writer import write_tei


@dataclass
class ArticleResult:
    article: str
    pdf: str
    ok: bool
    seconds: float
    tei_path: str | None = None
    n_references: int = 0
    n_consolidated: int = 0
    error: str | None = None


@dataclass
class RunSummary:
    backend: str
    model: str
    data_dir: str
    n_articles: int
    n_ok: int
    n_failed: int
    total_seconds: float
    results: list[ArticleResult] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2)


def run_extraction(
    data_dir: Path,
    backend_kind: str,
    config: BackendConfig,
    limit: int | None = None,
    overwrite: bool = False,
    progress=None,
    tei_suffix: str = LLM_TEI_SUFFIX,
    glutton_url: str = "",
    consolidate_citations: int = 0,
    consolidate_header: int = 0,
) -> RunSummary:
    data_dir = Path(data_dir)
    backend = get_backend(backend_kind, config)
    pdfs = find_pdfs(data_dir)
    if limit is not None:
        pdfs = pdfs[:limit]

    # Optional glutton consolidation, symmetric to GROBID's consolidateHeader/consolidateCitations.
    glutton = (
        GluttonClient(glutton_url)
        if glutton_url and (consolidate_citations or consolidate_header)
        else None
    )

    results: list[ArticleResult] = []
    run_start = time.time()

    for pdf in pdfs:
        article = pdf.parent.name
        expected_tei = pdf.with_name(pdf.stem + tei_suffix)
        if expected_tei.exists() and not overwrite:
            results.append(
                ArticleResult(
                    article=article,
                    pdf=str(pdf),
                    ok=True,
                    seconds=0.0,
                    tei_path=str(expected_tei),
                    n_references=0,
                    error="skipped (exists)",
                )
            )
            if progress:
                progress(article, True)
            continue

        start = time.time()
        try:
            extraction = backend.extract(pdf)
            n_consolidated = 0
            if glutton is not None:
                cons = consolidate_extraction(
                    extraction,
                    glutton,
                    citations_mode=consolidate_citations,
                    header_mode=consolidate_header,
                )
                n_consolidated = cons.references_matched
            tei_path = write_tei(extraction, pdf, suffix=tei_suffix)
            elapsed = time.time() - start
            results.append(
                ArticleResult(
                    article=article,
                    pdf=str(pdf),
                    ok=True,
                    seconds=round(elapsed, 2),
                    tei_path=str(tei_path),
                    n_references=len(extraction.references),
                    n_consolidated=n_consolidated,
                )
            )
            if progress:
                progress(article, True)
        except Exception as e:  # noqa: BLE001 - record and continue the batch
            elapsed = time.time() - start
            results.append(
                ArticleResult(
                    article=article,
                    pdf=str(pdf),
                    ok=False,
                    seconds=round(elapsed, 2),
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
                )
            )
            if progress:
                progress(article, False)

    if glutton is not None:
        glutton.close()

    total = time.time() - run_start
    n_ok = sum(1 for r in results if r.ok)
    return RunSummary(
        backend=backend.name,
        model=config.model,
        data_dir=str(data_dir),
        n_articles=len(results),
        n_ok=n_ok,
        n_failed=len(results) - n_ok,
        total_seconds=round(total, 2),
        results=results,
    )
