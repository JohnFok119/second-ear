# Phase 8 C2/C3 notes

## C2: what the dev picker chose, and why it is not "fewer guesses"
The picker (`scripts/pick_locata_decode.py`, grid in `results/where_binaural/locata_decode_dev.json`) maximises
pooled full 3-D F20 on LOCATA dev over thresholds 0.15-0.60 and caps 1, 2, 3, 5. It chose **threshold 0.15, cap 5**:
more guesses, not fewer (dev F20 0.127 at 4.73 guesses per real sound; cap 1 never beats 0.060 on dev).

Plain reading: on real recordings the model often puts a sound in the wrong front/back half or at the wrong height.
More guesses per frame give the matching more chances to land one guess near the truth, so full F20 goes up while
lateral F20 goes down (0.479 -> 0.321 on eval: more extra guesses on the left/right axis). The jump in
front/back-half correct (0.604 -> 0.907) has the same cause: with 5 guesses spread around, the matched one is
usually in the right half. It is not a sign the model learned front vs back. One guess per frame (cap 1) is the
honest "one answer" decode, and it scores worse on full F20. 0.15 is the lowest threshold in the grid.

## C3: harness checks
- Start point reproduces R0 on dev before any step: `--check-start` pooled dev F20 0.1015 at threshold 0.30, cap 5
  (`checkpoints/where_binaural_ft_startcheck/start_check.json`, git-ignored); R0 dev F20 0.101
  (`results/where_binaural_r0_locata_dev/metrics.json`).
- Overfit one recording (`locata_dev_task1_rec4`, all layers, lr 3e-4, 15 epochs, cap 5): train loss 0.00158 -> 0.00029,
  that recording's F20 0.193 (epoch 0) -> 0.800 (best, epoch 9, threshold 0.40). Log: `results/phase8/overfit_check.log`.
- Tests: `tests/test_finetune.py` (split has no overlap, frozen layers unchanged after a step, harness decode equals
  `BinauralHeatmapLocalizer.predict` on one dev recording).
