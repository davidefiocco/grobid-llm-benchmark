"""Deterministic backend for offline pipeline testing without a real LLM.

Derives a trivial extraction from the PDF's own text layer so the full
runner -> tei_writer -> scoring path can be exercised with no network or credentials.
"""

from __future__ import annotations

from pathlib import Path

from grobid_llm_benchmark.backends.base import LLMBackend
from grobid_llm_benchmark.models import Author, Body, Extraction, Header, Reference
from grobid_llm_benchmark.pdf import extract_text


class MockBackend(LLMBackend):
    @property
    def name(self) -> str:
        return f"mock:{self.config.model}"

    def extract(self, pdf_path: Path) -> Extraction:
        text = extract_text(pdf_path, max_pages=1)
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        title = lines[0] if lines else pdf_path.stem
        return Extraction(
            header=Header(
                title=title,
                authors=[Author(forename="Jane", surname="Doe")],
                abstract=" ".join(lines[1:4]),
                keywords=["mock"],
            ),
            references=[
                Reference(
                    title="A mock reference",
                    authors=[Author(surname="Smith")],
                    date="2020",
                    in_title="Journal of Mocking",
                    volume="1",
                    first_page="1",
                )
            ],
            body=Body(
                section_titles=["Introduction", "Methods"],
                figure_titles=["Figure 1"],
                table_titles=["Table 1"],
                citation_markers=["[1]"],
                figure_markers=["Figure 1"],
                table_markers=["Table 1"],
            ),
        )
