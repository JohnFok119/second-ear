# C5.1: generate the v2 binaural scene set under the cpu lease. Run in a herdr pane.
Set-Location C:\Users\johnn\personal\second-ear
$lease = "C:\Users\johnn\personal\fleet\lease.mjs"
while ($true) {
    node $lease request cpu --holder second-ear --what "C5.1 generate 4,600 binaural scenes" --peak "~6 cores, ~2 GB RAM"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 120
}
uv run python scripts/generate_binaural.py --split test --n-scenes 300 --workers 6 --out data/binaural_v2/test 2>&1 | Tee-Object -FilePath runs/c5_gen_test.log
uv run python scripts/generate_binaural.py --split train --n-scenes 300 --start-seed 100000 --workers 6 --out data/binaural_v2/val 2>&1 | Tee-Object -FilePath runs/c5_gen_val.log
uv run python scripts/generate_binaural.py --split train --n-scenes 4000 --workers 6 --out data/binaural_v2/train 2>&1 | Tee-Object -FilePath runs/c5_gen_train.log
"C5_DATA_DONE" | Out-File -FilePath runs/c5_data.done
node $lease release cpu --holder second-ear
