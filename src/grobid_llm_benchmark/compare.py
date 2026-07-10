"""Parse GROBID's markdown evaluation reports and build a GROBID-vs-LLM comparison.

GROBID's evaluation writes a ``report.md`` containing sections (Header metadata,
Citation metadata, ...) each with several markdown tables of the form::

    | label            |  precision |   recall  |     f1     | support |
    |---               |---         |---        |---         |---      |
    | title            | 95.2       | 93.1      | 94.1       | 20      |
    ...
    | all fields (micro avg) | ... |

Because we run the *same* evaluation for GROBID and for the LLM, both reports share the
identical structure. We extract the per-field f1 (and precision/recall) and lay them out
side by side. We key rows by (section heading, matching-mode heading, label) so the two
reports align exactly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_H_ANY = re.compile(r"^(#{2,6})\s+(.*)$")
_ROW = re.compile(r"^\|(.+)\|\s*$")


@dataclass
class Metric:
    precision: float | None
    recall: float | None
    f1: float | None
    support: str


# (context heading trail, field label) -> Metric
Key = tuple[str, str]
Metrics = dict[Key, Metric]


def _num(cell: str) -> float | None:
    cell = cell.strip()
    try:
        return float(cell)
    except ValueError:
        return None


def parse_report(path: Path) -> dict[tuple[str, str], Metric]:
    """Return {(context, label): Metric} where context is the full heading trail.

    GROBID's report reuses the same matching-mode subheadings (Strict/Soft/Levenshtein/
    Ratcliff) under *both* the "Header metadata" and "Citation metadata" sections, so the
    context must include the enclosing section heading — otherwise e.g. the header ``title``
    row and the citation ``title`` row collide and overwrite each other.
    """
    metrics: dict[tuple[str, str], Metric] = {}
    # heading trail keyed by depth: {level: text}; context joins levels in order.
    headings: dict[int, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        h = _H_ANY.match(line.strip())
        if h:
            level = len(h.group(1))
            headings[level] = h.group(2).strip()
            # drop any deeper headings now out of scope
            for deeper in [lvl for lvl in headings if lvl > level]:
                del headings[deeper]
            continue
        m = _ROW.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if len(cells) < 4:
            continue
        label = cells[0]
        if not label or set(label) <= {"-"} or label.lower() == "label":
            continue  # header/separator rows
        context = " / ".join(headings[lvl] for lvl in sorted(headings))
        metrics[(context, label)] = Metric(
            precision=_num(cells[1]),
            recall=_num(cells[2]),
            f1=_num(cells[3]),
            support=cells[4] if len(cells) > 4 else "",
        )
    return metrics


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value}"


def _pr(metric: Metric | None) -> str:
    return f"{metric.precision}/{metric.recall}" if metric else ""


def _delta(llm_f1: float | None, grobid_f1: float | None) -> str:
    if llm_f1 is None or grobid_f1 is None:
        return ""
    return f"{llm_f1 - grobid_f1:+.1f}"


_INTRO = (
    "GROBID and the LLM are peer tools: both are scored by GROBID's own end-to-end "
    "evaluation against the *same* gold data (same XPath fields, same matching modes), so "
    "their f1 values are directly comparable and gold is the only ground truth. `Δ` is the "
    "LLM's f1 minus GROBID's (positive = LLM better)."
)


def _sections(keys: Sequence[Key]) -> list[str]:
    """Distinct top-level GROBID sections present (e.g. ``Header metadata``), in first-seen order.

    Each key's context is a ``"<section> / <matching mode>"`` trail; the section is the part
    before the first ``/``. Listing them makes the report state *what* was actually scored
    (header only, +citations, +fulltext structures, ...), which varies by run.
    """
    seen: list[str] = []
    for ctx, _ in keys:
        section = ctx.split(" / ", 1)[0].strip()
        if section and section not in seen:
            seen.append(section)
    return seen


def _scope_line(labels: Sequence[str], keys: Sequence[Key]) -> str:
    """A one-line provenance/scope note tailored to this specific comparison."""
    backends = ", ".join(f"**{lbl}**" for lbl in labels)
    sections = ", ".join(_sections(keys)) or "—"
    return f"Comparing **GROBID** against {backends}. Sections scored: {sections}."


def build_comparison(
    grobid_report: Path,
    llm_reports: Sequence[tuple[str, Path]],
) -> str:
    """Build a GROBID-vs-LLM comparison from one GROBID report and one or more LLM reports.

    ``llm_reports`` is a sequence of ``(label, report_path)`` pairs. With a single LLM this
    emits a rich pairwise table treating GROBID and the LLM as two peer baselines (f1 **and**
    precision/recall for each, plus their f1 delta). With several LLMs it emits a compact
    overview: GROBID's f1 plus an ``f1`` + ``Δ`` column pair per backend, all sharing the one
    GROBID column.
    """
    g = parse_report(grobid_report)
    parsed = [(label, parse_report(path)) for label, path in llm_reports]

    keys = set(g)
    for _, metrics in parsed:
        keys |= set(metrics)
    keys = sorted(keys)

    if len(parsed) == 1:
        return _pairwise_table(g, parsed[0], keys)
    return _overview_table(g, parsed, keys)


def _pairwise_table(g: Metrics, llm: tuple[str, Metrics], keys: list[Key]) -> str:
    label, lm_all = llm
    lines = [
        f"# GROBID vs {label} — side-by-side evaluation",
        "",
        _scope_line([label], keys),
        "",
        _INTRO,
        "",
        f"| section / mode | field | GROBID f1 | {label} f1 | Δ | GROBID P/R | {label} P/R |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for ctx, field in keys:
        gm = g.get((ctx, field))
        lm = lm_all.get((ctx, field))
        gf = gm.f1 if gm else None
        lf = lm.f1 if lm else None
        lines.append(
            f"| {ctx} | {field} | {_fmt(gf)} | {_fmt(lf)} | {_delta(lf, gf)} | "
            f"{_pr(gm)} | {_pr(lm)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _overview_table(g: Metrics, parsed: list[tuple[str, Metrics]], keys: list[Key]) -> str:
    header = "| section / mode | field | GROBID f1 |"
    sep = "|---|---|---:|"
    for label, _ in parsed:
        header += f" {label} f1 | {label} Δ |"
        sep += "---:|---:|"

    labels = [label for label, _ in parsed]
    lines = [
        f"# GROBID vs {len(labels)} LLM backends — side-by-side evaluation",
        "",
        _scope_line(labels, keys),
        "",
        _INTRO + " Each backend is compared pairwise against the shared GROBID column.",
        "",
        header,
        sep,
    ]
    for ctx, field in keys:
        gm = g.get((ctx, field))
        gf = gm.f1 if gm else None
        row = f"| {ctx} | {field} | {_fmt(gf)} |"
        for _, metrics in parsed:
            lm = metrics.get((ctx, field))
            lf = lm.f1 if lm else None
            row += f" {_fmt(lf)} | {_delta(lf, gf)} |"
        lines.append(row)
    lines.append("")
    return "\n".join(lines)
