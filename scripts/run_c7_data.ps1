# phase7-C7: generate the v3 binaural scene set (48 kHz, real rooms, beds) and its features under the cpu lease.
Set-Location C:\Users\johnn\personal\second-ear
$lease = "C:\Users\johnn\personal\fleet\lease.mjs"
while ($true) {
    node $lease request cpu --holder second-ear --what "phase7-C7 generate 4,600 v3 scenes + features" --peak "~6 cores, ~3 GB RAM"
    if ($LASTEXITCODE -eq 0) { break }
    Start-Sleep -Seconds 120
}
uv run python scripts/generate_binaural.py --preset v3 --split test --n-scenes 300 --workers 6 --out data/binaural_v3/test 2>&1 | Tee-Object -FilePath runs/c7_gen_test.log
uv run python scripts/generate_binaural.py --preset v3 --split train --n-scenes 300 --start-seed 100000 --workers 6 --out data/binaural_v3/val 2>&1 | Tee-Object -FilePath runs/c7_gen_val.log
uv run python scripts/generate_binaural.py --preset v3 --split train --n-scenes 4000 --workers 6 --out data/binaural_v3/train 2>&1 | Tee-Object -FilePath runs/c7_gen_train.log
uv run python scripts/extract_binaural_features.py --v2 --root data/binaural_v3 --out data/features/binaural_v3 2>&1 | Tee-Object -FilePath runs/c7_feat.log
"C7_DATA_DONE" | Out-File -FilePath runs/c7_data.done
node $lease release cpu --holder second-ear
