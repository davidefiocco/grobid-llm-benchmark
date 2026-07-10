#!/bin/bash
# Configure GROBID consolidation to use biblio-glutton at $GLUTTON_URL, then start the
# service. Reference consolidation is what brings citation-field accuracy up to the
# published benchmark levels. Uses awk (not python3) so it works on the crf-lite image,
# which ships no python3.
set -euo pipefail

CONFIG=/opt/grobid/grobid-home/config/grobid.yaml
GLUTTON_URL="${GLUTTON_URL:-http://glutton:8080}"

if [ -f "$CONFIG" ]; then
  # Switch the consolidation service to glutton and point its url at $GLUTTON_URL. GROBID's
  # yaml keeps a "service:" key plus a nested glutton "url:" (with commented examples we must
  # skip). Best-effort: never block startup on a config-format quirk.
  awk -v url="$GLUTTON_URL" '
    !svc && /^[[:space:]]*service:[[:space:]]*"?(crossref|glutton|none)"?/ {
      sub(/service:.*/, "service: \"glutton\""); svc=1; print; next
    }
    /^[[:space:]]*glutton:[[:space:]]*$/ { inglut=1; print; next }
    inglut && !urldone && /^[[:space:]]*url:/ {
      sub(/url:.*/, "url: \"" url "\""); urldone=1; print; next
    }
    inglut && /^[[:space:]]*(crossref|proxy):/ { inglut=0 }
    { print }
  ' "$CONFIG" > "$CONFIG.tmp" && mv "$CONFIG.tmp" "$CONFIG" \
    && echo "grobid.yaml: consolidation=glutton url=$GLUTTON_URL" \
    || echo "grobid.yaml: consolidation rewrite skipped (kept defaults)"
fi

exec ./grobid-service/bin/grobid-service
