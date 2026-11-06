#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip

python3 -m pip install \
  "sphinx>=9.2.0,<9.3.0" \
  "sphinx-autoapi>=3.9.0,<3.10.0" \
  "furo==2025.12.19"

python3 -m sphinx -b html docs _site
