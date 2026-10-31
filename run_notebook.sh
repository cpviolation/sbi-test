#!/usr/bin/env bash
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

echo "=== Executing notebook (this will take ~8-10 min) ==="
uv run jupyter nbconvert \
    --to notebook \
    --execute \
    --inplace \
    --ExecutePreprocessor.timeout=1800 \
    notebooks/efficiency_dp_sbi.ipynb

echo "=== Notebook executed. Committing ==="
git -c user.email="maurizio.martinelli@unimib.it" \
    -c user.name="Maurizio Martinelli" \
    add notebooks/efficiency_dp_sbi.ipynb

git -c user.email="maurizio.martinelli@unimib.it" \
    -c user.name="Maurizio Martinelli" \
    commit -m "run: execute notebook — add cell outputs

Re-run after performance fixes:
  - N_EVT = N_DATA = 20 000 (removes max-normalisation bias)
  - N_SIM = 10 000 / 30 000 (1D/2D)
  - max_num_epochs = 200, stop_after_epochs = 20
  - GPU auto-select (≥20 GB VRAM threshold)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01PuGMZocPf8w3feAg8hBvYe"

echo "=== Pushing to origin/main ==="
git push origin main

echo "=== Done! ==="
