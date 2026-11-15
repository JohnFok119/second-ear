#!/usr/bin/env bash
# phase8-C1: single-source scoring of the v2 where_* models on the binaural_v2 test split (val-picked thresholds).
set -e
mkdir -p runs
uv run python scripts/eval_heatmap.py --config configs/where_binaural.yaml --threshold 0.30 --single-source --device cuda --out results/where_binaural_single > runs/p8_c1_binaural.log 2>&1
uv run python scripts/eval_heatmap.py --config configs/where_binaural.yaml --threshold 0.30 --mute left --single-source --device cuda --out results/where_binaural_mute_left_single > runs/p8_c1_muteleft.log 2>&1
uv run python scripts/eval_heatmap.py --config configs/where_right.yaml --threshold 0.20 --single-source --device cuda --out results/where_right_single > runs/p8_c1_right.log 2>&1
uv run python scripts/eval_heatmap.py --config configs/where_left.yaml --threshold 0.20 --single-source --device cuda --out results/where_left_single > runs/p8_c1_left.log 2>&1
echo ALLDONE
