"""Hosted-API multimodal backend for Azure OpenAI and OpenAI-compatible endpoints.

Renders PDF pages to images, sends them with an extraction prompt to a chat-completions
endpoint requesting a JSON object, and validates the result against the Pydantic schema.

Configuration is read from the environment at construction time (never baked into an
image):

- ``provider="azure"``: ``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``,
  ``AZURE_OPENAI_API_VERSION`` (and the deployment name is the configured model).
- ``provider="openai"``: ``OPENAI_API_KEY`` and optional ``OPENAI_BASE_URL`` (point this
  at any OpenAI-compatible server, e.g. an Ollama ``/v1`` endpoint or a gateway).
"""

from __future__ import annotations

import os
from pathlib import Path

from grobid_llm_benchmark.backends.base import BackendConfig, LLMBackend
from grobid_llm_benchmark.json_utils import coerce_json_ex
from grobid_llm_benchmark.models import EXTRACTION_JSON_HINT, EXTRACTION_TASK, Extraction
from grobid_llm_benchmark.pdf import extract_text, render_pages_to_base64

_SYSTEM = (
    "You are an information extraction system for scholarly articles. "
    "You read the page images (and any provided text) of a scientific PDF and output "
    "structured bibliographic and full-text data. Be exhaustive: cover every reference "
    "and every section, figure, table and in-text call-out across the whole document."
)

# Text-layer budget. Large enough to reach the bibliography at the end of a long article
# (the first ~20k chars only cover the front matter); still bounded for outlier documents.
_MAX_TEXT_CHARS = 120_000


class ApiBackend(LLMBackend):
    """Chat-completions backend for Azure OpenAI or an OpenAI-compatible endpoint."""

    def __init__(self, config: BackendConfig, provider: str = "azure"):
        super().__init__(config)
        self.provider = provider
        self._client = self._make_client(provider, config)

    @staticmethod
    def _make_client(provider: str, config: BackendConfig):
        timeout = config.timeout or None
        if provider == "azure":
            from openai import AzureOpenAI

            endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set")
            return AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                timeout=timeout,
            )
        from openai import OpenAI

        return OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.config.model}"

    def _build_content(self, pdf_path: Path) -> list[dict]:
        cfg = self.config
        content: list[dict] = [
            {
                "type": "text",
                "text": EXTRACTION_TASK + "\n\n" + EXTRACTION_JSON_HINT,
            }
        ]
        if cfg.include_text:
            text = extract_text(pdf_path, max_pages=cfg.max_pages)
            if text.strip():
                content.append(
                    {
                        "type": "text",
                        "text": "Extracted text layer of the article:\n\n" + text[:_MAX_TEXT_CHARS],
                    }
                )
        if cfg.include_images:
            for b64 in render_pages_to_base64(pdf_path, max_pages=cfg.max_pages, dpi=cfg.dpi):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
        return content

    def extract(self, pdf_path: Path) -> Extraction:
        cfg = self.config
        response = self._client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": self._build_content(pdf_path)},
            ],
            temperature=cfg.temperature,
            max_tokens=cfg.num_predict or None,
            response_format={"type": "json_object"},
        )
        self.last_truncated = getattr(response.choices[0], "finish_reason", None) == "length"
        content = response.choices[0].message.content or ""
        try:
            return Extraction.model_validate_json(content)
        except Exception:
            obj, repaired = coerce_json_ex(content)
            self.last_truncated = self.last_truncated or repaired
            return Extraction.model_validate(obj)
