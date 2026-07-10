"""Score TEI against gold NLM using GROBID's own end-to-end evaluation.

Delegates to the patched grobid checkout's Gradle tasks so the numbers use GROBID's exact
XPath fields and matching modes:

- ``jatsEval  -Prun=0``  scores GROBID's ``*.fulltext.tei.xml`` (baseline).
- ``jatsEvalLLM``        scores the LLM's ``*.fulltext.llm.tei.xml``.

Both write the same ``grobid-home/tmp/report.md``, which we copy to a caller-chosen path.
Scoring must therefore run serially (the benchmark pipeline does): concurrent invocations
would race on that shared file before it is copied.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from grobid_llm_benchmark.tei_schema import LLM_TEI_SUFFIX

_REPORT_REL = "grobid-home/tmp/report.md"


def _run_gradle(grobid_dir: Path, task: str, data_dir: Path, extra: list[str]) -> None:
    cmd = [
        "./gradlew",
        task,
        f"-Pp2t={Path(data_dir).resolve()}",
        "--console=plain",
        "--no-daemon",
        *extra,
    ]
    subprocess.run(cmd, cwd=grobid_dir, check=True)


def score(
    which: str,
    data_dir: Path,
    grobid_dir: Path,
    report_out: Path,
    llm_suffix: str = LLM_TEI_SUFFIX,
) -> Path:
    """Run a scoring task ('grobid' or 'llm') and copy its report to ``report_out``.

    'grobid' scores pre-produced ``*.fulltext.tei.xml`` (``jatsEval -Prun=0``); 'llm'
    scores the TEI matching ``llm_suffix`` (``jatsEvalLLM -Pllmsuffix=...``), defaulting to
    ``*.fulltext.llm.tei.xml``. Pass a backend-tagged suffix to score one specific backend
    when several coexist in the dataset dir. GROBID PDF processing is expected to have
    already run via the GROBID service.
    """
    grobid_dir = Path(grobid_dir)
    if which == "grobid":
        _run_gradle(grobid_dir, "jatsEval", data_dir, ["-Prun=0"])
    elif which == "llm":
        _run_gradle(grobid_dir, "jatsEvalLLM", data_dir, [f"-Pllmsuffix={llm_suffix}"])
    else:
        raise ValueError(f"which must be 'grobid' or 'llm', got {which!r}")

    produced = grobid_dir / _REPORT_REL
    if not produced.exists():
        raise FileNotFoundError(f"expected report at {produced}")
    report_out = Path(report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(produced, report_out)
    return report_out
