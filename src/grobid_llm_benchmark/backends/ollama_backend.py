"""Ollama multimodal backend.

Sends rendered PDF page images (and optionally the extracted text layer) to an Ollama
vision model and requests the structured extraction JSON, constraining the output to the
Pydantic schema via Ollama's ``format`` argument.
"""

from __future__ import annotations

from pathlib import Path

import ollama

from grobid_llm_benchmark.backends.base import LLMBackend
from grobid_llm_benchmark.json_utils import coerce_json_ex
from grobid_llm_benchmark.models import EXTRACTION_JSON_HINT, EXTRACTION_TASK, Extraction
from grobid_llm_benchmark.pdf import extract_text, render_pages_to_png

_SYSTEM = (
    "You are an information extraction system for scholarly articles. "
    "You read the page images (and any provided text) of a scientific PDF and output "
    "structured bibliographic and full-text data. Be exhaustive: cover every reference "
    "and every section, figure, table and in-text call-out across the whole document."
)


class OllamaBackend(LLMBackend):
    @property
    def name(self) -> str:
        return f"ollama:{self.config.model}"

    def extract(self, pdf_path: Path) -> Extraction:
        cfg = self.config
        images = (
            render_pages_to_png(pdf_path, max_pages=cfg.max_pages, dpi=cfg.dpi)
            if cfg.include_images
            else []
        )

        prompt_parts = [EXTRACTION_TASK, EXTRACTION_JSON_HINT]
        if cfg.include_text:
            text = extract_text(pdf_path, max_pages=cfg.max_pages)
            if text.strip():
                prompt_parts.append(
                    "For convenience, here is the extracted (possibly noisy) text layer "
                    "of the article:\n\n" + text[:20000]
                )
        prompt = "\n\n".join(prompt_parts)

        options = {"temperature": cfg.temperature}
        if cfg.num_ctx:
            options["num_ctx"] = cfg.num_ctx
        if cfg.num_predict:
            options["num_predict"] = cfg.num_predict

        user_msg = {"role": "user", "content": prompt}
        if images:
            user_msg["images"] = images
        response = ollama.chat(
            model=cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                user_msg,
            ],
            format=Extraction.model_json_schema(),
            options=options,
        )
        self.last_truncated = response.get("done_reason") == "length"
        content = response["message"]["content"]
        try:
            return Extraction.model_validate_json(content)
        except Exception:
            obj, repaired = coerce_json_ex(content)
            self.last_truncated = self.last_truncated or repaired
            return Extraction.model_validate(obj)
