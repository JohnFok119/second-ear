# Train the where-only models on free Google Colab

Colab gives you a free GPU (a T4) for a few hours at a time. The notebook picks up where your local training stopped and saves progress to Google Drive, so a disconnect only costs you a re-run.

## One-time setup (on your PC)

1. Pack the features into small half-size (float16) files:
   `uv run python scripts/pack_features_f16.py --src data/features/binaural_v2 --out C:\temp\feat_f16 --splits train val --shard-gb 1.0`
   It refuses to start if the disk is too full. Do this when no training run is using the machine.
2. In Google Drive, make the folder `MyDrive/second-ear/`.
3. Upload the whole `feat_f16` shard folder contents to `MyDrive/second-ear/features/` (files named `feat_f16_000.tar`, `feat_f16_001.tar`, ...).
4. Optional, only to score on Colab: tar the raw val and test scenes (audio + labels) and upload the tar next to the shards:
   `tar -cf scenes_v3.tar -C data/binaural_v3 val test`, then put `scenes_v3.tar` in `MyDrive/second-ear/features/`. Any file named `scenes_*.tar` there is unpacked into `data/<DATA>/`.
5. Upload these 4 files from `checkpoints/where_binaural/` to `MyDrive/second-ear/checkpoints/where_binaural/`: `last.pt`, `best.pt`, `history.jsonl`, `scaler.npz`.

## Run it

1. Open the notebook in Colab: File > Open notebook > GitHub tab > `JohnFok119/second-ear`, branch `phase7`, file `colab/train_where.ipynb`. (Or upload `colab/train_where.ipynb` by hand.) If the repo is private, use the Upload tab.
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > **Run all**. Allow Drive access when asked.

Two settings in the Settings cell choose the data: `DATA` is the feature set name (default `binaural_v2`, use `binaural_v3` for the new scenes) and `FEATURES` is the Drive subfolder that holds the shards and scenes tars (default `features`).

It trains `binaural`, then `left`, then `right`. A model that finished gets a `done.txt` in its checkpoint folder and is skipped next time.

## Score on Colab (last cell)

The last cell, "7. Score", checks each finished model (the ones with a `done.txt`). If it has no `test_done.txt` yet, the cell picks the best threshold on the val scenes, then tests on the test scenes (the two-ear model also with the left and right ear muted). The small result folders (`results/<experiment>`, plus the `_thr...` and `_mute_...` ones) are copied to `MyDrive/second-ear/results/`, and `test_done.txt` marks the model as scored. It needs the `scenes_*.tar` from setup step 4. Without the val scenes it skips the model; without the test scenes it only picks the threshold. If `MyDrive/second-ear/locata/locata_scenes.tar` exists (members `scenes/dev/*` and `scenes/eval/*`), the cell also scores each two-ear model on the real LOCATA recordings with the picked threshold and marks it with `locata_done.txt`; each part is skipped only by its own marker, so older models still get their LOCATA run on a later pass.

## After a disconnect

Open the same notebook, pick a T4 again and **Run all**. It re-downloads the code, unpacks the features (about a few minutes), and resumes each unfinished model from its `last.pt` on Drive. History is appended, not overwritten.

## Get the results back

1. In Drive, download `MyDrive/second-ear/checkpoints/` (all `where_*` folders) and copy them into the local `checkpoints/` folder, replacing the older files.
2. Optional: copy `MyDrive/second-ear/runs/` into the local `runs/` folder (TensorBoard logs and `where_<model>_train.log`).

3. If you scored on Colab, copy `MyDrive/second-ear/results/` into the local `results/` folder. Otherwise threshold picking, test evaluation and mutes run locally. ITD and export always run locally.
