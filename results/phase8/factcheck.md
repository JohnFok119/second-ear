# Phase 8 C5 fact-check (crew-reviewer, 2026-11-15)

About 190 numbers in `verdict.md`, README (Phase 8 subsections) and LEARNING (Phase 8) checked against the metrics
JSON files, `recipe_pick.json`, `ft_runs/*`, `locata_decode_dev.json` and the LOCATA label CSVs.
Two rounding mismatches found and fixed: right-ear ratio 1.66x -> 1.65x (mean over seeds), two-ear T2 ratio
2.05x -> 2.04x (raw values). One loose wording fixed in LEARNING ("about 5x" -> "about 5.5x").
Not checked against a metrics file: the "4,000 training scenes" count (older README text, unchanged here).
Result after fixes: clean.
