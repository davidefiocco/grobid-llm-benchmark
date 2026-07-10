"""API backend with a stubbed client: no real Azure/OpenAI call, no key.

Verifies the backend builds multimodal content and maps a JSON response into an
``Extraction`` without touching the network.
"""

from types import SimpleNamespace

import pytest

from grobid_llm_benchmark.backends.api_backend import ApiBackend
from grobid_llm_benchmark.backends.base import BackendConfig

pytestmark = pytest.mark.offline

_JSON = (
    '{"header": {"title": "A Paper", "authors": [{"forename": "J", "surname": "Doe"}], '
    '"abstract": "abs", "keywords": ["k"]}, '
    '"references": [{"title": "R1", "authors": [{"surname": "Smith"}], "date": "2011"}]}'
)


class _StubClient:
    def __init__(self):
        self.captured = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        msg = SimpleNamespace(content=_JSON)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_api_backend_parses_response(fixtures_dir, monkeypatch):
    stub = _StubClient()
    monkeypatch.setattr(ApiBackend, "_make_client", staticmethod(lambda provider, config: stub))

    cfg = BackendConfig(model="gpt-4o", max_pages=1, include_images=True, include_text=False)
    backend = ApiBackend(cfg, provider="azure")

    pdf = next(fixtures_dir.rglob("*.pdf"))
    extraction = backend.extract(pdf)

    assert extraction.header.title == "A Paper"
    assert extraction.references[0].title == "R1"
    # request asked for a JSON object and sent the page image
    assert stub.captured["response_format"] == {"type": "json_object"}
    content = stub.captured["messages"][1]["content"]
    assert any(part["type"] == "image_url" for part in content)


def test_api_backend_name():
    cfg = BackendConfig(model="gpt-4o")
    # name must not touch the client; patch construction to a no-op
    backend = ApiBackend.__new__(ApiBackend)
    backend.config = cfg
    backend.provider = "azure"
    assert backend.name == "azure:gpt-4o"
