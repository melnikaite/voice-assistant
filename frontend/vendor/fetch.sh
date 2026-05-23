#!/usr/bin/env bash
# Vendor the third-party browser libraries into ./vendor so the assistant
# stays usable when the host has no internet.  Run once at setup time
# (and again whenever you bump versions).  index.html already references
# these local paths — until this script has run the UI will 404 on them.
#
# Pinning: we pin to a specific @<version> instead of @latest so that a
# silent CDN-side update can't change behaviour out from under us.
# Update by editing the URLs below and rerunning.
set -euo pipefail

cd "$(dirname "$0")"

# Pico CSS v2.x (2.0.6 is the latest at time of vendoring).  ~20 KB
# minified.  All the design tokens, no JS.
PICO_URL="https://cdn.jsdelivr.net/npm/@picocss/pico@2.0.6/css/pico.min.css"
PICO_DST="pico.min.css"

# Chart.js v4 (UMD build — sufficient for our use as a <script> tag).
# ~200 KB minified.  We only use Chart() + a few default scales, no
# plugins.
CHART_URL="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"
CHART_DST="chart.umd.min.js"

echo "Fetching Pico CSS …"
curl -sSfL "$PICO_URL" -o "$PICO_DST"
echo "  → $PICO_DST ($(wc -c < "$PICO_DST") bytes)"

echo "Fetching Chart.js …"
curl -sSfL "$CHART_URL" -o "$CHART_DST"
echo "  → $CHART_DST ($(wc -c < "$CHART_DST") bytes)"

echo "Done.  The frontend now works fully offline."
