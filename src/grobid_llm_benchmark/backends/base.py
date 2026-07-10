"""Backend interface: turn a PDF into a structured :class:`Extraction`.

Backends are thin and swappable behind a common interface. Each receives the article PDF
path plus a shared config and returns an ``Extraction`` (or raises); the runner times each
call and records failures.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

from grobid_llm_benchmark.models import Extraction


@dataclass
class BackendConfig:
    """Shared backend configuration. Defaults mirror the ``glb run`` CLI so constructing
    ``BackendConfig`` directly (tests, scripts) matches the benchmark configuration."""

    model: str
    # 0 => send every page (whole-document coverage), matching the CLI default
    max_pages: int = 0
    dpi: int = 150
    # feed the PDF text layer alongside images (helps with references)
    include_text: bool = True
    # send rendered page images (disable for text-only models)
    include_images: bool = True
    temperature: float = 0.0
    # per-request timeout in seconds (0 => no client-side timeout)
    timeout: float = 600.0
    # context window (Ollama only); 0 => server default
    num_ctx: int = 8192
    # cap generated tokens (reference lists can be long, but avoid runaway generation)
    num_predict: int = 5000


class LLMBackend(abc.ABC):
    """Abstract multimodal extraction backend."""

    def __init__(self, config: BackendConfig):
        self.config = config
        # Set by extract() when the JSON response had to be repaired from a truncated
        # (token-capped) generation, so the runner can flag a possibly-cut reference list.
        self.last_truncated = False

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def extract(self, pdf_path: Path) -> Extraction:
        """Extract header + references from a single article PDF."""


def get_backend(kind: str, config: BackendConfig) -> LLMBackend:
    """Resolve a backend by name."""
    kind = kind.lower()
    if kind in ("azure", "openai"):
        from grobid_llm_benchmark.backends.api_backend import ApiBackend

        return ApiBackend(config, provider=kind)
    if kind == "ollama":
        from grobid_llm_benchmark.backends.ollama_backend import OllamaBackend

        return OllamaBackend(config)
    if kind == "mock":
        from grobid_llm_benchmark.backends.mock_backend import MockBackend

        return MockBackend(config)
    raise ValueError(f"Unknown backend '{kind}'. Available: azure, openai, ollama, mock")
