# Train the where-only models on free Google Colab

Colab gives you a free GPU (a T4) for a few hours at a time. The notebook picks up where your local training stopped and saves progress to Google Drive, so a disconnect only costs you a re-run.

## One-time setup (on your PC)

1. Pack the features into small half-size (float16) files:
   `uv run python scripts/pack_features_f16.py --src data/features/binaural_v2 --out C:\temp\feat_f16 --splits train val --shard-gb 1.0`
   It refuses to start if the disk is too full. Do this when no training run is using the machine.
2. In Google Drive, make the folder `MyDrive/second-ear/`.
3. Upload the whole `feat_f16` shard folder contents to `MyDrive/second-ear/features/` (files named `feat_f16_000.tar`, `feat_f16_001.tar`, ...).
4. Upload these 4 files from `checkpoints/where_binaural/` to `MyDrive/second-ear/checkpoints/where_binaural/`: `last.pt`, `best.pt`, `history.jsonl`, `scaler.npz`.

## Run it

1. Open the notebook in Colab: File > Open notebook > GitHub tab > `JohnFok119/second-ear`, branch `feature/colab-train`, file `colab/train_where.ipynb`. (Or upload `colab/train_where.ipynb` by hand.) If the repo is private, use the Upload tab.
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > **Run all**. Allow Drive access when asked.

It trains `binaural`, then `left`, then `right`. A model that finished gets a `done.txt` in its checkpoint folder and is skipped next time.

## After a disconnect

Open the same notebook, pick a T4 again and **Run all**. It re-downloads the code, unpacks the features (about a few minutes), and resumes each unfinished model from its `last.pt` on Drive. History is appended, not overwritten.

## Get the results back

1. In Drive, download `MyDrive/second-ear/checkpoints/` (all `where_*` folders) and copy them into the local `checkpoints/` folder, replacing the older files.
2. Optional: copy `MyDrive/second-ear/runs/` into the local `runs/` folder (TensorBoard logs and `where_<model>_train.log`).

Threshold picking, test evaluation, mutes, ITD and export still run locally, because they need the raw audio.
