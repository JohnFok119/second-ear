# Vendored DCASE 2023 SELD baseline

Source: https://github.com/sharathadavanne/seld-dcase2023 at commit
`24d14c9ebe6878062dc3b68eb07dd0100722f530`. MIT License, (c) Sharath Adavanne /
DCASE 2023 baseline authors (see `LICENSE`). No behaviour changes to the model or metrics.

Every vendored `.py` carries `# ruff: noqa` and `# fmt: off` so upstream formatting stays untouched.

## Changes made to upstream code

- `seldnet_model.py`: removed `from IPython import embed`. Nothing else.
- `SELD_evaluation_metrics.py`: removed `from IPython import embed`; `np.finfo(np.float).eps`
  became `np.finfo(float).eps` (`np.float` was removed in numpy 1.24).
- `parameters.py`: removed all `print(...)` calls; an unknown id now raises `ValueError` instead of
  `print` + `exit()`; `if argv == '1':` got a `pass` body (its only statement was a print); removed the
  empty trailing `for key, value in params.items():` loop (its body was a print). Param values are untouched.
- `cls_feature_class.py`: trimmed to the FOA parts. Kept verbatim: `nCr`, `_next_greater_power_of_2`,
  `_spectrogram`, `_get_mel_spectrogram`, `_get_foa_intensity_vectors`, `get_adpit_labels_for_file`,
  `segment_labels`, `convert_output_format_polar_to_cartesian`, `convert_output_format_cartesian_to_polar`,
  and the small getters. `__init__` keeps only the signal/mel/label-rate fields (no directories, no
  SALSA-lite/GCC branch). Dropped: sklearn/joblib/matplotlib/shutil/wave imports, audio/label file IO,
  GCC, SALSA-lite (where the `np.int`/`np.float` calls lived), normalisation, batch extraction.
  `exit()` on NaN features became `raise ValueError`.
- `decoding.py` (new file, extracted from `train_seldnet.py`): `get_multi_accdoa_labels` and
  `determine_similar_location` verbatim; the multi-ACCDOA branch of `test_epoch()` moved into
  `multi_accdoa_to_output_dict()` (the three duplicated "lone track" branches folded into one parameterised
  branch, same arithmetic). Import of `distance_between_cartesian_coordinates` made relative.
- Not vendored: `cls_compute_seld_results.py` (directory-walking wrapper; `eval/official.py` calls
  `SELDMetrics` directly), `cls_data_generator.py`, `batch_feature_extraction.py`, the rest of `train_seldnet.py`.
