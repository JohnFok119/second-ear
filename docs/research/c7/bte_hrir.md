# BTE HRIR datasets (phase7-C13)

Researched 2026-11-02 (crew-researcher, research only, nothing downloaded).

Purpose: our model trains on in-ear HRIRs, but the real LOCATA test set (Zenodo 3630471) is recorded with
hearing-aid mics behind the ear (BTE). Left/right works on LOCATA; front/back and elevation fail
(`results/phase7/locata_diagnosis.md`). BTE-position HRIRs would let us render training data that matches.

Licence rule: CC0 / CC BY / MIT / ODC-By; CC BY-SA only for data that is never committed; no NC, ND, research-only.

| dataset | mic positions | directions | distance | rooms | rate | size | licence (exact) | licence URL | download | usable? |
|---|---|---|---|---|---|---|---|---|---|---|
| MMHR-HRTF (Thiemann, van de Par), SOFA v1.3.1 | 4 HATS with in-ear mics; B&K 4128C ("BKwHA") has 6 extra BTE-aid channels (3 mics per side, per secondary source) | 2 deg az/el sphere, lowest el -64 deg; 12,722 directions (secondary source) | 1.7 m | anechoic (not stated on fetched pages) | unknown (not stated on Zenodo or README); 44.1 kHz only inferred (4410 samples = 100 ms on the uol.de page) | BKwHA.sofa 3.15 GB; BKwHA_s.sofa 212.8 MB; record total 5.8 GB | "Creative Commons Attribution 4.0 International"; README: "published under the Creative Commons (CC-BY 4.0) license" | https://zenodo.org/records/3767044 | https://zenodo.org/records/3767044/files/BKwHA_s.sofa?download=1 | **yes** |
| MMHR-HRTF v1.3 (older) | same | same | same | same | unknown | same | CC BY 4.0 | https://zenodo.org/records/2582553 | superseded by 3767044 | yes, but use v1.3.1 (azimuth sign fixed) |
| Kayser 2009 Oldenburg BTE + in-ear HRIR/BRIR | 3-ch BTE aids + in-ear mic per ear (8 ch) | not stated on fetched pages | not stated | anechoic + 4 real rooms | unknown | unknown | "Permission to use this database for purely research or educational purposes is granted. No commercial exploitation of this database is permitted." | https://medi.uni-oldenburg.de/hrir/html/download.html | name/email form, link by email | **no** (research-only, form) |
| OlHeaD-HRTF (Denk 2018) | eardrum + 13 mic locations in 6 hearing-device styles | 91 directions; 16 subjects + 3 dummy heads | unknown | unknown | unknown | unknown | "Creative Commons BY-NC-SA 4.0 license" | https://uol.de/p53485 | licence click-through | **no** (NC, SA) |
| Hearpiece DB (Denk) | in-the-ear earpiece, not BTE | 87 directions | unknown | unknown | unknown | unknown | CC BY-SA 4.0 (search result only, record not fetched) | https://zenodo.org/record/3733190 (not fetched) | n/a | no (not BTE; SA) |
| Clarity challenge HRIRs | BTE front/mid/rear mics per Clarity docs | unknown | unknown | unknown | unknown | HRIRs_MAT 167 MB | unknown (not stated on any page read) | https://claritychallenge.github.io/docs/cec2/data/cec2_data (search snippet only) | Clarity download page | unknown; source not confirmed |

## LOCATA hearing aids

LOCATA paper (arXiv 1909.01008, search snippet only): a pair of non-commercial Siemens Signia Pure 7mi dummies on a
HEAD acoustics HMS II head-torso simulator; each aid has 2 Sonion 50GC30-MP2 mics 9 mm apart (4 channels total);
aids 157 mm apart. Zenodo record: "Hearing aid dummies on a dummy head (2-channel per hearing aid)", licence
"Open Data Commons Attribution License v1.0".

No dataset matches the hardware exactly. MMHR has an HMSII.2 head but with in-ear mics only; its BTE aid is a
Siemens Acuris dummy (3 mics per side) on a B&K 4128C.

## Recommendation

Use **MMHR-HRTF v1.3.1** (https://zenodo.org/records/3767044), file **BKwHA_s.sofa** (212.8 MB, truncated to
6.66 ms) plus README.txt. Needs Johnny's OK to download. Before rendering, read the channel order and sample rate
from the SOFA file and pick the front BTE mic per ear. Close to LOCATA (BTE aid, same kind of 2-mic-per-side
geometry) but not the same hardware.

## Unknowns

- MMHR sample rate and channel order (not in README; SOFA not opened).
- Whether MMHR was measured anechoic (not stated on fetched pages).
- Clarity HRIR source and licence.
- Hearpiece licence (search snippet only).
- Exact LOCATA BTE mic coordinates (LOCATA documentation PDF not opened:
  https://www.locata.lms.tf.fau.de/files/2020/01/Documentation_LOCATA_final_release_V1.pdf).
- Broader searches found no other BTE sets.
