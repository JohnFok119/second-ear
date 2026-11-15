#!/usr/bin/env bash
# phase8-C4: fine-tune runs on LOCATA dev (3 seeds per row), each scored on LOCATA eval and the binaural_v2 test split.
# R2 two ears head only (lr 1e-3); R3 two ears all layers (lr 1e-4); R4 right ear / R5 left ear with the recipe
# that has the higher mean fine-tune-val F20 (dev only, never eval). Cap 5 for every run (R0/R1 and the dev pick).
set -e
mkdir -p runs
SEEDS="0 1 2"
EPOCHS=30
run_one () {  # config mode lr seed
  local cfg=$1 mode=$2 lr=$3 seed=$4
  local exp; exp=$(basename "$cfg" .yaml)
  local name=${exp}_ft_${mode}_s${seed}
  local ck=checkpoints/$name
  if [ ! -f "$ck/threshold_val.json" ] || [ ! -f "$ck/done" ]; then
    uv run python scripts/finetune_locata.py --config "$cfg" --mode "$mode" --lr "$lr" --epochs $EPOCHS --seed "$seed" --cap 5 --device cuda --out-dir "$ck" > runs/p8_${name}.log 2>&1
    touch "$ck/done"
  fi
  local thr; thr=$(uv run python -c "import json;print(json.load(open('$ck/threshold_val.json'))['chosen'])")
  [ -f results/${name}_locata_eval/metrics.json ] || uv run python scripts/eval_locata.py --config "$cfg" --ckpt-dir "$ck" --threshold "$thr" --max-per-frame 5 --experiment-name "$name" --splits eval --device cuda > runs/p8_${name}_eval.log 2>&1
  [ -f results/${name}_sim/metrics.json ] || uv run python scripts/eval_heatmap.py --config "$cfg" --ckpt-dir "$ck" --threshold "$thr" --device cuda --out results/${name}_sim > runs/p8_${name}_sim.log 2>&1
  echo "done $name thr=$thr"
}
for s in $SEEDS; do run_one configs/where_binaural.yaml head 1e-3 $s; done
for s in $SEEDS; do run_one configs/where_binaural.yaml all 1e-4 $s; done
PICK=$(uv run python scripts/pick_ft_recipe.py)
echo "recipe: $PICK"
MODE=${PICK%% *}; LR=${PICK##* }
for s in $SEEDS; do run_one configs/where_right.yaml $MODE $LR $s; done
for s in $SEEDS; do run_one configs/where_left.yaml $MODE $LR $s; done
echo ALLDONE
