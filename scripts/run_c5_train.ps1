# C5.2 training chain: v2 feature cache, then where-only two-ear / left / right models.
# Best epoch and decoding threshold are picked on the val split; test is evaluated once at that threshold.
# Run in a herdr pane while holding the GPU lease. Modelled on run_c3_chain.ps1 + run_c3_valthr.ps1.
Set-Location C:\Users\johnn\personal\second-ear
# Features must already exist in data/features/binaural_v2 (built separately by extract_binaural_features.py --v2).
foreach ($e in @("binaural", "left", "right")) {
    uv run python scripts/train_heatmap.py --config configs/where_$e.yaml --device cuda 2>&1 | Tee-Object -FilePath runs/where_${e}_train.log
    uv run python scripts/pick_threshold.py --config configs/where_$e.yaml --split val --device cuda 2>&1 | Tee-Object -FilePath runs/where_${e}_pick_threshold.log
    $chosen = (Get-Content results/where_$e/threshold_val.json -Raw | ConvertFrom-Json).chosen
    uv run python scripts/eval_heatmap.py --config configs/where_$e.yaml --split test --device cuda --threshold $chosen 2>&1 | Tee-Object -FilePath runs/where_${e}_eval.log
    if ($e -eq "binaural") {
        foreach ($m in @("left", "right")) {
            uv run python scripts/eval_heatmap.py --config configs/where_binaural.yaml --split test --mute $m --device cuda --threshold $chosen 2>&1 | Tee-Object -FilePath runs/where_binaural_mute_${m}_eval.log
        }
    }
}
uv run python scripts/eval_itd.py --root data/binaural_v2 --split test --out results/where_classical_itd 2>&1 | Tee-Object -FilePath runs/where_classical_itd.log
uv run python scripts/one_ear_table.py --where 2>&1 | Tee-Object -FilePath runs/one_ear_table_where.log
"CHAIN_C5_TRAIN_DONE" | Out-File -FilePath runs/c5_train.done
node C:\Users\johnn\personal\fleet\lease.mjs release gpu --holder second-ear
