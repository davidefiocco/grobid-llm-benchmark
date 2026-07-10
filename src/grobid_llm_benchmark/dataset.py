"""Download GROBID's gold evaluation datasets from HuggingFace.

The dataset ``sciencialab/grobid-evaluation`` stores one directory per article, each
containing at least ``<name>.pdf`` and ``<name>.nxml`` (the gold JATS/NLM file). ``n``
selects how many article directories to fetch: the dataset's full size for the complete
benchmark (1943 for ``PMC_sample_1943``), or fewer for a quick sample.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "sciencialab/grobid-evaluation"

# Extensions we care about for the end-to-end evaluation. The gold file is the .nxml
# (PMC) / .xml (bioRxiv); the PDF is the input. Everything else (.pdfx.xml, an existing
# .fulltext.tei.xml) is skipped so the article dirs stay clean.
_PDF_SUFFIX = ".pdf"
_GOLD_SUFFIXES = (".nxml",)


def _group_by_article(files: list[str], dataset_dir: str) -> dict[str, list[str]]:
    """Group repo file paths by their article sub-directory within ``dataset_dir``."""
    groups: dict[str, list[str]] = defaultdict(list)
    prefix = dataset_dir.rstrip("/") + "/"
    for f in files:
        if not f.startswith(prefix):
            continue
        rest = f[len(prefix) :]
        if "/" not in rest:
            continue  # top-level file (e.g. .DS_Store)
        article = rest.split("/", 1)[0]
        groups[article].append(f)
    return groups


def download_slice(
    out_dir: Path,
    n: int = 1943,
    dataset_dir: str = "PMC_sample_1943",
    seed: int = 13,
) -> list[Path]:
    """Download ``n`` article directories from ``dataset_dir`` into ``out_dir``.

    Only the PDF and the gold ``.nxml`` are fetched for each article. Returns the list
    of local article directories that ended up with both a PDF and a gold file.
    """
    import random

    api = HfApi()
    all_files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    groups = _group_by_article(all_files, dataset_dir)

    # keep only articles that have both a pdf and a gold file
    usable: dict[str, tuple[str, str]] = {}
    for article, paths in groups.items():
        pdf = next((p for p in paths if p.lower().endswith(_PDF_SUFFIX)), None)
        gold = next(
            (p for p in paths if any(p.endswith(s) for s in _GOLD_SUFFIXES)),
            None,
        )
        if pdf and gold:
            usable[article] = (pdf, gold)

    articles = sorted(usable)
    random.Random(seed).shuffle(articles)
    selected = articles[:n]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_dirs: list[Path] = []
    for article in selected:
        pdf_repo, gold_repo = usable[article]
        article_dir = out_dir / article
        article_dir.mkdir(parents=True, exist_ok=True)
        for repo_path in (pdf_repo, gold_repo):
            cached = hf_hub_download(
                DATASET_REPO,
                repo_path,
                repo_type="dataset",
            )
            dest = article_dir / Path(repo_path).name
            shutil.copyfile(cached, dest)
        result_dirs.append(article_dir)

    return result_dirs


def find_article_dirs(data_dir: Path) -> list[Path]:
    """Return article sub-directories under ``data_dir`` that contain both a PDF and a gold file."""
    data_dir = Path(data_dir)
    dirs: list[Path] = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        has_pdf = any(p.suffix.lower() == ".pdf" for p in child.iterdir())
        has_gold = any(p.name.endswith(s) for p in child.iterdir() for s in _GOLD_SUFFIXES)
        if has_pdf and has_gold:
            dirs.append(child)
    return dirs
