"""Java scorer smoke test: score the LLM TEI fixture with GROBID's evaluation.

Runs ``jatsEvalLLM`` on a copy of the fixture (which carries a committed
``*.fulltext.llm.tei.xml``) and asserts a parseable ``report.md`` is produced. Requires a
JDK and the patched grobid checkout, so it skips when those are absent (e.g. on a host
without Java); in the harness image both are present.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from grobid_llm_benchmark.compare import parse_report

pytestmark = pytest.mark.docker

# In the harness container GROBID_DIR=/opt/grobid; locally it defaults to a sibling checkout.
_GROBID_DIR = Path(
    os.environ.get("GROBID_DIR", str(Path(__file__).parent.parent.parent / "grobid"))
)


def _have_java() -> bool:
    if not (_GROBID_DIR / "gradlew").exists() or shutil.which("java") is None:
        return False
    # macOS ships a `java` shim that exits non-zero when no JRE is installed.
    return subprocess.run(["java", "-version"], capture_output=True).returncode == 0


requires_java = pytest.mark.skipif(not _have_java(), reason="JDK / grobid checkout absent")


@requires_java
def test_llm_scorer_emits_parseable_report(pilot_dir, tmp_path):
    from grobid_llm_benchmark.scorer import score

    data = tmp_path / "data"
    shutil.copytree(pilot_dir, data)
    # the committed fixture lacks the LLM TEI (gitignored); generate one via the mock path
    from grobid_llm_benchmark.backends.base import BackendConfig
    from grobid_llm_benchmark.runner import run_extraction

    run_extraction(data, "mock", BackendConfig(model="mock", include_images=False))

    report = score("llm", data, _GROBID_DIR, tmp_path / "llm_report.md")
    metrics = parse_report(report)
    assert metrics, "scorer produced an empty/unparseable report"
