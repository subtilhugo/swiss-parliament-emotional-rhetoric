#!/usr/bin/env bash
# Full pipeline: fetch → label → build → compile
# Set OPENAI_API_KEY before running (or add it to a .env file).
#
# Steps 00-04 are expensive (~18 h, ~$30 at gpt-4o-mini prices for 110k speeches).
# Skip them to use the committed frozen dataset instead (steps 05-06 only).
#
# Usage:
#   ./run_all.sh            # full end-to-end
#   ./run_all.sh --skip-api # use frozen data, just rebuild charts and PDF
set -euo pipefail

SKIP_API=false
for arg in "$@"; do
  [[ "$arg" == "--skip-api" ]] && SKIP_API=true
done

if [[ "$SKIP_API" == false ]]; then
  echo "=== 00 Fetch speeches from parliament API ==="
  python3 src/00_fetch.py

  echo "=== 01 Label rhetoric (reason / emotion / neutral) ==="
  python3 src/01_label_rhetoric.py

  echo "=== 02 Label emotions (8-emotion scores) ==="
  python3 src/02_label_emotions.py

  echo "=== 03 Label CAP topics ==="
  python3 src/03_label_topics.py

  echo "=== 04 Merge labels into final dataset ==="
  python3 src/04_merge.py
else
  echo "=== Skipping API steps — using frozen dataset ==="
fi

echo "=== 05 Build chart-level CSVs ==="
python3 src/build.py

echo "=== 06 Compile PDF ==="
python3 src/compile.py

echo "Done. PDF: slides_letemps_parlacap.pdf"
