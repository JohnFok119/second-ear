#!/usr/bin/env bash
# phase8-C2 + C3 sanity: R0 (as is) and R1 (dev-picked decoding) on LOCATA, start check, overfit check.
set -e
mkdir -p runs
C=configs/where_binaural.yaml
uv run python scripts/eval_locata.py --config $C --threshold 0.30 --max-per-frame 5 --tag r0 --splits dev eval --device cuda > runs/p8_r0.log 2>&1
uv run python scripts/pick_locata_decode.py --config $C --split dev --device cuda > runs/p8_pick.log 2>&1
THR=$(uv run python -c "import json;print(json.load(open('results/where_binaural/locata_decode_dev.json'))['chosen']['threshold'])")
CAP=$(uv run python -c "import json;print(json.load(open('results/where_binaural/locata_decode_dev.json'))['chosen']['cap'])")
echo "picked thr=$THR cap=$CAP"
uv run python scripts/eval_locata.py --config $C --threshold $THR --max-per-frame $CAP --tag r1 --splits dev eval --device cuda > runs/p8_r1.log 2>&1
uv run python scripts/finetune_locata.py --config $C --mode all --lr 1e-4 --cap 5 --thresholds 0.30 --check-start --device cuda --out-dir checkpoints/where_binaural_ft_startcheck > runs/p8_startcheck.log 2>&1
uv run python scripts/finetune_locata.py --config $C --mode all --lr 3e-4 --epochs 15 --cap 5 --overfit-one locata_dev_task1_rec4 --device cuda --out-dir checkpoints/where_binaural_ft_overfit1 > runs/p8_overfit.log 2>&1
echo ALLDONE
