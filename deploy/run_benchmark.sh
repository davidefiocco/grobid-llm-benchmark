#!/usr/bin/env bash
# One-shot GROBID-vs-LLM benchmark, run inside the harness container.
#
#   1. ensure the dataset                -> data/<dataset>/
#   2. GROBID baseline via the service   -> *.fulltext.tei.xml
#   3. score the GROBID baseline         -> reports/grobid_report.md
#   4. (BASELINE_ONLY=0) LLM extraction  -> *.fulltext.llm.tei.xml
#   5. (BASELINE_ONLY=0) score the LLM   -> reports/llm_report.md
#   6. (BASELINE_ONLY=0) comparison      -> reports/comparison_<tag>.md
#
# N_ARTICLES sizes the run: a small value for a quick sample, or the dataset's full size
# (1943 for PMC_sample_1943) for the complete published benchmark.
#
# Env: GROBID_URL, GROBID_DIR, N_ARTICLES, DATASET, BASELINE_ONLY, LLM_BACKEND, LLM_MODEL,
#      LLM_TAG (namespace the LLM TEI so backends coexist), LLM_REPORT (report filename),
#      CONSOLIDATE_HEADER, CONSOLIDATE_CITATIONS.
set -euo pipefail

DATASET="${DATASET:-PMC_sample_1943}"
DATA_DIR="${DATA_DIR:-/opt/harness/data/${DATASET}}"
REPORTS_DIR="${REPORTS_DIR:-/opt/harness/reports}"
# accept the legacy PILOT_N name; default is the full PMC_sample_1943 set.
N_ARTICLES="${N_ARTICLES:-${PILOT_N:-1943}}"
BASELINE_ONLY="${BASELINE_ONLY:-1}"
LLM_BACKEND="${LLM_BACKEND:-azure}"
LLM_MODEL="${LLM_MODEL:-gpt-4o}"
# Optional backend tag: namespaces the LLM TEI as *.fulltext.llm.<tag>.tei.xml so several
# backends can coexist in the same mounted data volume. Empty = canonical suffix.
LLM_TAG="${LLM_TAG:-}"
LLM_REPORT="${LLM_REPORT:-llm_report.md}"
# GROBID's published PMC benchmark runs consolidateHeader=1 + consolidateCitations=0 (see
# grobid-trainer EndToEndEvaluation). We mirror that: header consolidation on (needs glutton),
# citation consolidation OFF. Turning citation consolidation on replaces parsed reference fields
# with full-precision CrossRef metadata (e.g. a full date vs the year-only JATS gold), which the
# scorer compares verbatim and which deflates rather than lifts citation scores.
CONSOLIDATE_HEADER="${CONSOLIDATE_HEADER:-1}"
CONSOLIDATE_CITATIONS="${CONSOLIDATE_CITATIONS:-0}"
# For a symmetric comparison the LLM uses the *same* consolidation as GROBID by default (both get
# glutton header consolidation, neither gets citation consolidation). Override independently if you
# want to demo the LLM's DOI-enrichment path. Needs GLUTTON_URL reachable during the LLM run.
LLM_CONSOLIDATE_HEADER="${LLM_CONSOLIDATE_HEADER:-$CONSOLIDATE_HEADER}"
LLM_CONSOLIDATE_CITATIONS="${LLM_CONSOLIDATE_CITATIONS:-$CONSOLIDATE_CITATIONS}"

mkdir -p "$DATA_DIR" "$REPORTS_DIR"

echo "=== [1] dataset ($N_ARTICLES articles from $DATASET) at $DATA_DIR ==="
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  glb download-data --out "$DATA_DIR" --dataset "$DATASET" --n "$N_ARTICLES"
else
  echo "  dataset present, skipping download"
fi

echo "=== [2] GROBID baseline via service -> *.fulltext.tei.xml ==="
glb grobid-run --data "$DATA_DIR" \
  --consolidate-header "$CONSOLIDATE_HEADER" \
  --consolidate-citations "$CONSOLIDATE_CITATIONS"

echo "=== [3] score GROBID baseline -> grobid_report.md ==="
glb score grobid --data "$DATA_DIR" --grobid-dir "$GROBID_DIR" \
  --out "$REPORTS_DIR/grobid_report.md"

if [ "$BASELINE_ONLY" = "1" ]; then
  echo "=== BASELINE_ONLY: skipping LLM run/score/compare. ==="
  echo "=== done. GROBID baseline report at $REPORTS_DIR/grobid_report.md ==="
  exit 0
fi

echo "=== [4] LLM extraction ($LLM_BACKEND:$LLM_MODEL, tag='${LLM_TAG:-<none>}') -> *.fulltext.llm[.tag].tei.xml ==="
glb run --data "$DATA_DIR" --backend "$LLM_BACKEND" --model "$LLM_MODEL" --tag "$LLM_TAG" \
  --glutton-url "${GLUTTON_URL:-}" \
  --consolidate-header "$LLM_CONSOLIDATE_HEADER" \
  --consolidate-citations "$LLM_CONSOLIDATE_CITATIONS" \
  --summary-out "$REPORTS_DIR/llm_run.json"

echo "=== [5] score LLM -> $LLM_REPORT ==="
glb score llm --data "$DATA_DIR" --grobid-dir "$GROBID_DIR" --tag "$LLM_TAG" \
  --out "$REPORTS_DIR/$LLM_REPORT"

# Name the comparison after the backend so distinct runs don't clobber one shared file.
COMPARISON_REPORT="comparison_${LLM_TAG:-llm}.md"
echo "=== [6] comparison -> $COMPARISON_REPORT ==="
glb compare \
  --grobid-report "$REPORTS_DIR/grobid_report.md" \
  --llm-report "${LLM_TAG:-llm}=$REPORTS_DIR/$LLM_REPORT" \
  --out "$REPORTS_DIR/$COMPARISON_REPORT"

echo "=== done. artifacts in $REPORTS_DIR ==="
