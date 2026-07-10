"""Offline end-to-end: run the pilot with the mock backend and validate the TEI.

Exercises dataset discovery -> backend -> tei_writer -> file output with no network, GPU,
or LLM credentials, proving the harness wiring is intact.
"""

import shutil

import pytest
from lxml import etree

from grobid_llm_benchmark import tei_schema as S
from grobid_llm_benchmark.backends.base import BackendConfig
from grobid_llm_benchmark.dataset import find_article_dirs
from grobid_llm_benchmark.runner import run_extraction

pytestmark = pytest.mark.offline

NS = {"tei": S.TEI_NS}


@pytest.fixture
def work_dir(pilot_dir, tmp_path):
    """A writable copy of the fixture dataset so generated TEI does not touch fixtures/."""
    dest = tmp_path / "data"
    shutil.copytree(pilot_dir, dest)
    return dest


def test_mock_pilot_writes_valid_tei(work_dir):
    cfg = BackendConfig(model="mock", include_images=False, include_text=True)
    summary = run_extraction(work_dir, "mock", cfg)

    assert summary.n_articles >= 1
    assert summary.n_ok == summary.n_articles
    assert summary.n_failed == 0

    for article_dir in find_article_dirs(work_dir):
        teis = list(article_dir.glob("*.fulltext.llm.tei.xml"))
        assert teis, f"no LLM TEI written in {article_dir}"
        root = etree.parse(str(teis[0])).getroot()
        # a header title and at least one reference are present in the TEI
        assert root.xpath("//tei:titleStmt/tei:title/text()", namespaces=NS)
        assert root.xpath("//tei:back//tei:biblStruct", namespaces=NS)
        # fulltext structures flow through to the body too
        assert root.xpath("//tei:text/tei:body/tei:div/tei:head/text()", namespaces=NS)
        assert root.xpath('//tei:ref[@type="bibr"]/text()', namespaces=NS)


def test_backend_tag_namespaces_tei_so_backends_coexist(work_dir):
    """Two tagged runs leave distinct TEI side by side, neither clobbering the other."""
    cfg = BackendConfig(model="mock", include_images=False, include_text=True)
    run_extraction(work_dir, "mock", cfg, tei_suffix=S.llm_tei_suffix("azure"))
    run_extraction(work_dir, "mock", cfg, tei_suffix=S.llm_tei_suffix("ollama"))

    for article_dir in find_article_dirs(work_dir):
        assert list(article_dir.glob("*.fulltext.llm.azure.tei.xml"))
        assert list(article_dir.glob("*.fulltext.llm.ollama.tei.xml"))
        # a tagged run must not also emit the canonical (untagged) suffix
        assert not list(article_dir.glob("*.fulltext.llm.tei.xml"))
