# C5.2 eval chain after Colab training: same steps as run_c5_train.ps1 minus training.
# Needs checkpoints/where_{binaural,left,right} copied back from Drive. Run in a herdr pane; takes the gpu lease itself.
param([string[]]$Models = @("binaural", "left", "right"))
$Models = $Models | ForEach-Object { $_ -split "," } | Where-Object { $_ }  # -File passes "a,b" as one string
Set-Location C:\Users\johnn\personal\second-ear
$lease = "C:\Users\johnn\personal\fleet\lease.mjs"
while ($true) {
    node $lease request gpu --holder second-ear --what "C5.2 eval: val threshold + test eval, 3 where-only models" --peak "~3 GB VRAM, ~3 GB RAM"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 120
}
foreach ($e in $Models) {
    uv run python scripts/pick_threshold.py --config configs/where_$e.yaml --split val --device cuda 2>&1 | Tee-Object -FilePath runs/where_${e}_pick_threshold.log
    $chosen = (Get-Content results/where_$e/threshold_val.json -Raw | ConvertFrom-Json).chosen
    uv run python scripts/eval_heatmap.py --config configs/where_$e.yaml --split test --device cuda --threshold $chosen 2>&1 | Tee-Object -FilePath runs/where_${e}_eval.log
    if ($e -eq "binaural") {
        foreach ($m in @("left", "right")) {
            uv run python scripts/eval_heatmap.py --config configs/where_binaural.yaml --split test --mute $m --device cuda --threshold $chosen 2>&1 | Tee-Object -FilePath runs/where_binaural_mute_${m}_eval.log
        }
    }
}
if ($Models -contains "right") {
uv run python scripts/eval_itd.py --root data/binaural_v2 --split test --out results/where_classical_itd 2>&1 | Tee-Object -FilePath runs/where_classical_itd.log
uv run python scripts/one_ear_table.py --where 2>&1 | Tee-Object -FilePath runs/one_ear_table_where.log
"CHAIN_C5_EVAL_DONE" | Out-File -FilePath runs/c5_eval.done
}
node $lease release gpu --holder second-ear
