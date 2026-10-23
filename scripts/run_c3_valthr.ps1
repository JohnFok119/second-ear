# C3.3 follow-up: pick the decoding threshold on validation, evaluate test at it, re-export, rebuild the table.
# Run in a herdr pane while holding the GPU lease, after run_c3_chain.ps1 has finished.
Set-Location C:\Users\johnn\personal\second-ear
foreach ($e in @("binaural", "left", "right")) {
    uv run python scripts/pick_threshold.py --config configs/heatmap_$e.yaml --device cuda 2>&1 | Tee-Object -FilePath runs/pick_threshold_$e.log
    $chosen = (Get-Content results/heatmap_$e/threshold_val.json -Raw | ConvertFrom-Json).chosen
    uv run python scripts/eval_heatmap.py --config configs/heatmap_$e.yaml --device cuda --threshold $chosen 2>&1 | Tee-Object -FilePath runs/heatmap_${e}_eval_thr.log
    if ($e -eq "binaural") {
        foreach ($m in @("left", "right")) {
            uv run python scripts/eval_heatmap.py --config configs/heatmap_binaural.yaml --mute $m --device cuda --threshold $chosen 2>&1 | Tee-Object -FilePath runs/heatmap_binaural_mute_${m}_eval_thr.log
        }
    }
}
uv run python scripts/export_coffee_shop.py --device cuda --val-threshold 2>&1 | Tee-Object -FilePath runs/export_coffee_shop_valthr.log
uv run python scripts/one_ear_table.py --val-threshold 2>&1 | Tee-Object -FilePath runs/one_ear_table_valthr.log
"CHAIN_VALTHR_DONE" | Out-File -FilePath runs/chain_valthr.done
node C:\Users\johnn\personal\fleet\lease.mjs release gpu --holder second-ear
