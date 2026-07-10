"""Pluggable LLM backends for the benchmark harness."""

from grobid_llm_benchmark.backends.base import LLMBackend, get_backend

__all__ = ["LLMBackend", "get_backend"]
