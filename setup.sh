#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt && pip install -e . && { [ -f .env ] || cp .env.example .env; } && pre-commit install
