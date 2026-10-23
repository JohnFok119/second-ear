# C3.2/C3.3 training chain (C2.4 FOA eval done on CPU 2026-10-23). Run in a herdr pane while holding the GPU lease.
Set-Location C:\Users\johnn\personal\second-ear
foreach ($e in @("binaural", "left", "right")) {
    uv run python scripts/train_heatmap.py --config configs/heatmap_$e.yaml --device cuda 2>&1 | Tee-Object -FilePath runs/heatmap_${e}_train.log
    uv run python scripts/eval_heatmap.py --config configs/heatmap_$e.yaml --device cuda 2>&1 | Tee-Object -FilePath runs/heatmap_${e}_eval.log
    if ($e -eq "binaural") {
        foreach ($m in @("left", "right")) {
            uv run python scripts/eval_heatmap.py --config configs/heatmap_binaural.yaml --mute $m --device cuda 2>&1 | Tee-Object -FilePath runs/heatmap_binaural_mute_${m}_eval.log
        }
    }
}
uv run python scripts/export_coffee_shop.py --device cuda 2>&1 | Tee-Object -FilePath runs/export_coffee_shop.log
uv run python scripts/one_ear_table.py 2>&1 | Tee-Object -FilePath runs/one_ear_table.log
"CHAIN_C3_DONE" | Out-File -FilePath runs/chain_c3.done
node C:\Users\johnn\personal
node C:\Users\johnn\personal\fleet\lease.mjs release gpu --holder second-ear
