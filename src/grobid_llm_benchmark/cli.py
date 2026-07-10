"""Typer CLI for the GROBID-vs-LLM benchmark harness."""

from __future__ import annotations

import os
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from grobid_llm_benchmark.backends.base import BackendConfig
from grobid_llm_benchmark.compare import build_comparison
from grobid_llm_benchmark.cost import RATES, estimate
from grobid_llm_benchmark.dataset import download_slice, find_article_dirs
from grobid_llm_benchmark.pdf import find_pdfs
from grobid_llm_benchmark.runner import run_extraction
from grobid_llm_benchmark.tei_schema import llm_tei_suffix

app = typer.Typer(
    add_completion=False, help="Benchmark multimodal LLMs against GROBID (issue #1146)."
)
console = Console()


@app.command("download-data")
def download_data(
    out: Path = typer.Option(Path("./data/PMC_sample_1943"), help="Output directory."),
    n: int = typer.Option(
        1943, help="Number of article directories to fetch (dataset size for a full run)."
    ),
    dataset: str = typer.Option("PMC_sample_1943", help="Dataset dir in the HF repo."),
    seed: int = typer.Option(
        13, help="Shuffle seed for reproducible selection when n < dataset size."
    ),
):
    """Download a GROBID gold dataset (PDF + gold .nxml per article); n=dataset size for a full run."""
    dirs = download_slice(out, n=n, dataset_dir=dataset, seed=seed)
    console.print(f"[green]Downloaded {len(dirs)} article directories to {out}[/green]")


@app.command("run")
def run(
    data: Path = typer.Option(..., help="Dataset dir with per-article sub-directories."),
    backend: str = typer.Option("azure", help="LLM backend: azure, openai, ollama, mock."),
    model: str = typer.Option(
        lambda: os.environ.get("LLM_MODEL", "gpt-4o"), help="Model / deployment name."
    ),
    max_pages: int = typer.Option(
        0, help="Max PDF pages to send as images (first+last); 0 = all pages (parity coverage)."
    ),
    dpi: int = typer.Option(110, help="Render DPI for page images."),
    include_text: bool = typer.Option(False, help="Also send the PDF text layer."),
    num_ctx: int = typer.Option(6144, help="Context window (Ollama only; small avoids swap)."),
    num_predict: int = typer.Option(5000, help="Max generated tokens."),
    limit: int = typer.Option(0, help="Limit number of articles (0 = all)."),
    overwrite: bool = typer.Option(False, help="Re-run even if a TEI already exists."),
    tag: str = typer.Option(
        "",
        help="Backend tag namespacing the TEI (e.g. 'azure' -> *.fulltext.llm.azure.tei.xml) "
        "so multiple backends can coexist; score with the same --tag.",
    ),
    glutton_url: str = typer.Option(
        lambda: os.environ.get("GLUTTON_URL", ""),
        help="biblio-glutton base URL for optional consolidation (empty = no consolidation).",
    ),
    consolidate_citations: int = typer.Option(
        0,
        help="Consolidate references via glutton (0=off, 1=replace fields, 2=identifiers only). "
        "Like GROBID, mode 1 pulls full-precision CrossRef dates, so keep it off the scored "
        "citation comparison.",
    ),
    consolidate_header: int = typer.Option(
        0, help="Consolidate header via glutton (0=off, 1=replace fields). Safe to score."
    ),
    summary_out: Path = typer.Option(Path("./reports/llm_run.json"), help="Run summary JSON."),
):
    """Run the LLM over each PDF and write the TEI (default *.fulltext.llm.tei.xml) next to it."""
    cfg = BackendConfig(
        model=model,
        max_pages=max_pages,
        dpi=dpi,
        include_text=include_text,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    suffix = llm_tei_suffix(tag)
    n_dirs = len(find_pdfs(data))
    console.print(
        f"Running [bold]{backend}:{model}[/bold] over {n_dirs} articles in {data} "
        f"(TEI suffix [bold]{suffix}[/bold])"
    )

    done = {"n": 0}

    def progress(name: str, ok: bool):
        done["n"] += 1
        status = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  [{done['n']}/{n_dirs}] {name}: {status}")

    summary = run_extraction(
        data,
        backend,
        cfg,
        limit=limit or None,
        overwrite=overwrite,
        progress=progress,
        tei_suffix=suffix,
        glutton_url=glutton_url,
        consolidate_citations=consolidate_citations,
        consolidate_header=consolidate_header,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(summary.to_json())

    table = Table(title="LLM run summary")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("backend", summary.backend)
    table.add_row("articles", str(summary.n_articles))
    table.add_row("ok", str(summary.n_ok))
    table.add_row("failed", str(summary.n_failed))
    n_truncated = sum(1 for r in summary.results if r.truncated)
    if n_truncated:
        table.add_row("truncated (refs may be cut)", str(n_truncated))
    table.add_row("total seconds", f"{summary.total_seconds:.1f}")
    if summary.n_ok:
        avg = summary.total_seconds / max(summary.n_ok, 1)
        table.add_row("avg s/article", f"{avg:.1f}")
    console.print(table)
    console.print(f"[green]Summary written to {summary_out}[/green]")


@app.command("grobid-run")
def grobid_run(
    data: Path = typer.Option(..., help="Dataset dir with per-article sub-directories."),
    server: str = typer.Option(
        lambda: os.environ.get("GROBID_URL", "http://localhost:8070"),
        help="GROBID REST server base URL.",
    ),
    consolidate_citations: int = typer.Option(
        0,
        help=(
            "GROBID consolidateCitations (0=off, 1=replace fields via glutton/crossref, "
            "2=DOI-only). Default 0 to match GROBID's published benchmark flow; enabling it "
            "replaces parsed reference fields with full-precision CrossRef metadata that "
            "mismatches the year-granularity JATS gold and deflates citation scores."
        ),
    ),
    consolidate_header: int = typer.Option(
        1, help="GROBID consolidateHeader (1=via glutton/crossref, 0=off). Needs a service."
    ),
    timeout: float = typer.Option(300.0, help="Per-PDF request timeout (seconds)."),
    limit: int = typer.Option(0, help="Limit number of articles (0 = all)."),
    overwrite: bool = typer.Option(False, help="Re-run even if a TEI already exists."),
):
    """Produce GROBID *.fulltext.tei.xml for each PDF via the running GROBID service."""
    from grobid_llm_benchmark.grobid_client import is_alive, run_grobid

    if not is_alive(server):
        raise typer.BadParameter(f"GROBID service not alive at {server}")
    n_dirs = len(find_article_dirs(data))
    console.print(f"Running GROBID at [bold]{server}[/bold] over {n_dirs} articles in {data}")

    done = {"n": 0}

    def progress(name: str, ok: bool):
        done["n"] += 1
        status = "[green]ok[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  [{done['n']}/{n_dirs}] {name}: {status}")

    summary = run_grobid(
        data,
        server,
        timeout=timeout,
        consolidate_citations=consolidate_citations,
        consolidate_header=consolidate_header,
        overwrite=overwrite,
        limit=limit or None,
        progress=progress,
    )
    console.print(
        f"[green]GROBID: {summary.n_ok} ok, {summary.n_failed} failed "
        f"in {summary.total_seconds:.1f}s[/green]"
    )
    for f in summary.failures:
        console.print(f"  [red]{f}[/red]")


@app.command("estimate-cost")
def estimate_cost(
    n: int = typer.Option(20, help="Number of articles in the run."),
    rate: str = typer.Option("sonnet", help=f"Rate preset: {', '.join(RATES)}."),
    max_pages: int = typer.Option(4, help="Pages sent per article."),
    dpi: int = typer.Option(110, help="Render DPI."),
    output_tokens: int = typer.Option(3000, help="Assumed output tokens per article."),
):
    """Estimate the API dollar cost of a run before spending anything."""
    if rate not in RATES:
        raise typer.BadParameter(f"Unknown rate '{rate}'. Options: {', '.join(RATES)}")
    est = estimate(n, RATES[rate], max_pages=max_pages, dpi=dpi, output_tokens=output_tokens)
    table = Table(title=f"Estimated cost ({rate}, {n} articles)")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in ("input_tokens_per_article", "output_tokens_per_article"):
        table.add_row(key, str(est[key]))
    table.add_row("input cost (USD)", f"${est['input_cost_usd']:.2f}")
    table.add_row("output cost (USD)", f"${est['output_cost_usd']:.2f}")
    table.add_row("USD/article", f"${est['usd_per_article']:.4f}")
    table.add_row("total (USD)", f"${est['total_usd']:.2f}")
    console.print(table)


@app.command("score")
def score_cmd(
    which: str = typer.Argument(..., help="What to score: 'grobid' or 'llm'."),
    data: Path = typer.Option(..., help="Dataset dir with per-article sub-directories."),
    grobid_dir: Path = typer.Option(
        lambda: Path(os.environ.get("GROBID_DIR", "./grobid")),
        help="Patched grobid checkout with the jatsEval/jatsEvalLLM tasks "
        "(defaults to $GROBID_DIR or ./grobid).",
    ),
    out: Path = typer.Option(..., help="Where to copy the produced report.md."),
    tag: str = typer.Option(
        "",
        help="Backend tag to score (must match the --tag used at `run` time); "
        "ignored when scoring 'grobid'.",
    ),
):
    """Score TEI against gold with GROBID's evaluation (writes a report.md)."""
    from grobid_llm_benchmark.scorer import score

    report = score(which, data, grobid_dir, out, llm_suffix=llm_tei_suffix(tag))
    console.print(f"[green]{which} report written to {report}[/green]")


def _parse_llm_report_arg(item: str) -> tuple[str, Path]:
    """Parse a ``--llm-report`` value: ``label=path`` or just ``path`` (label from filename)."""
    if "=" in item:
        label, _, path = item.partition("=")
        path = path.strip()
        return (label.strip() or Path(path).stem), Path(path)
    stem = Path(item).stem
    if stem.startswith("llm_"):
        stem = stem[len("llm_") :]
    if stem.endswith("_report"):
        stem = stem[: -len("_report")]
    return (stem or "LLM"), Path(item)


def _default_comparison_out(entries: list[tuple[str, Path]]) -> Path:
    """Derive ``reports/comparison_<label...>.md`` from the backends being compared.

    Keeps each comparison's filename self-describing (e.g. ``comparison_gpt-4o-mini.md`` or
    ``comparison_azure_ollama.md``) so distinct runs don't overwrite one shared file.
    """
    slugs = [re.sub(r"[^A-Za-z0-9.-]+", "-", label).strip("-") or "llm" for label, _ in entries]
    return Path("./reports") / f"comparison_{'_'.join(slugs)}.md"


@app.command("compare")
def compare(
    grobid_report: Path = typer.Option(..., help="GROBID jatsEval report.md."),
    llm_report: list[str] = typer.Option(
        ...,
        "--llm-report",
        help="LLM jatsEvalLLM report.md; repeat for multiple backends. Optionally "
        "'label=path' to name the column (else derived from the filename), e.g. "
        "--llm-report azure=reports/llm_azure_report.md --llm-report ollama=reports/llm_ollama_report.md.",
    ),
    out: Path = typer.Option(
        None,
        help="Output comparison markdown. Default: reports/comparison_<backend...>.md, "
        "named after the compared backend(s).",
    ),
):
    """Build a GROBID-vs-LLM f1 comparison across one or more LLM reports (shared GROBID column)."""
    entries = [_parse_llm_report_arg(item) for item in llm_report]
    if out is None:
        out = _default_comparison_out(entries)
    md = build_comparison(grobid_report, entries)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    console.print(f"[green]Comparison written to {out}[/green]")
    console.print(md)


if __name__ == "__main__":
    app()
