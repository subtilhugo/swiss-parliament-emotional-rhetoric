#!/usr/bin/env bash
set -euo pipefail
python3 src/build.py
python3 src/compile.py
