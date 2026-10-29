# Phase 7 real-audio sources (phase7-C4)

Checked 2026-10-29. Files live in the MAIN repo `C:\Users\johnn\personal\second-ear\data\` (git-ignored, never committed). Licence rule: CC0, CC BY 3.0/4.0, MIT, ODC-By accepted; CC BY-SA only if nothing else covers the need (never commit/ship); NC, ND, Sampling+, research-only, unclear rejected.

Total on disk: BRIR 4.5 GB (cap 15 GB), beds 41 MB. Free space on C: 90 GB after downloads.

## BRIR (`data/brir/<set>/`)

All dummy-head (HATS/KEMAR) measurements. Direction grids are horizontal only (elevation 0), made by turning the head or moving the loudspeaker.

| Set | File | Licence as written on source page | Licence page | Rate | Ch | Directions / content | Rooms, RT60 | Dummy head |
|---|---|---|---|---|---|---|---|---|
| surrey_cortex | UniS_Anechoic_BRIR_48k.sofa | MIT, "Copyright (c) 2016 Institute of Sound Recording" (repo LICENSE; the Surrey IoSR software page links this repo as the official download) | https://github.com/IoSR-Surrey/RealRoomBRIRs/blob/master/LICENSE | 48 kHz | 2 | 37 azimuths, +-90 deg in 5 deg steps, el 0, loudspeaker at 1.5 m; IR 591 samples (pseudo-anechoic) | anechoic reference (truncated before first reflection) | Cortex Instruments Mk.2 HATS |
| surrey_cortex | UniS_Room_A_BRIR_48k.sofa | MIT (as above) | same | 48 kHz | 2 | 37 az, +-90 deg / 5 deg, el 0, 1.5 m; 18776 samples | Room A, office seating 8; RT60 0.32 s (measurement PDF Table 1) | Cortex Mk.2 HATS |
| surrey_cortex | UniS_Room_B_BRIR_48k.sofa | MIT | same | 48 kHz | 2 | 37 az, same grid; 48001 samples | Room B, small classroom; RT60 0.47 s | Cortex Mk.2 HATS |
| surrey_cortex | UniS_Room_C_BRIR_48k.sofa | MIT | same | 48 kHz | 2 | 37 az, same grid; 48001 samples | Room C, 418-seat lecture theatre; RT60 0.68 s | Cortex Mk.2 HATS |
| surrey_cortex | UniS_Room_D_BRIR_48k.sofa | MIT | same | 48 kHz | 2 | 37 az, same grid; 49047 samples | Room D, seminar space, high ceiling; RT60 0.89 s | Cortex Mk.2 HATS |
| spirit_tuberlin | QU_KEMAR_spirit.sofa | "Creative Commons (CC-BY-4.0)" (Zenodo record licence field cc-by-4.0; file metadata also CC BY 4.0) | https://zenodo.org/records/160751 | 44.1 kHz | 2 | 181 head orientations (-90..+90 deg, 1 deg) x 3 loudspeaker positions; IR 44100 samples | small meeting room "Spirit", TU Berlin; RT60 not stated on record | KEMAR 45BA, large pinnae |
| auditorium3_tuberlin | QU_KEMAR_Auditorium3.sofa | CC BY 4.0 (Zenodo licence field cc-by-4.0; file metadata CC BY 4.0) | https://zenodo.org/records/160749 | 44.1 kHz | 2 | 181 head orientations (-90..+90 deg, 1 deg) x 6 loudspeaker positions; 44100 samples | mid-size lecture hall Auditorium 3, TU Berlin; RT60 not stated on record | KEMAR 45BA, large pinnae |
| adream_laas | BRIRs.zip (5 SOFA: pos1-4, posall, trajectory + README) | "Licensed under Creative Commons (CC-BY-4.0)" (README in zip; Zenodo field cc-by-4.0; SOFA metadata CC BY 4.0) | https://zenodo.org/records/49357 | 44.1 kHz | 2 | 4 loudspeaker positions x 4 listener positions, head +-78 deg in 2 deg steps (79 orientations per file, checked on pos1); plus 20-position trajectory; 88200 samples | ADREAM apartment-like lab, LAAS-CNRS Toulouse; RT60 not stated on record | KEMAR 45BB-4, large pinnae |
| calypso (**HOLD, licence conflict**) | KEMAR_Calypso_Surround.sofa | Zenodo page: "Licensed under Creative Commons (CC-BY-4.0)", field cc-by-4.0. BUT the SOFA file's own `License` attribute says "Creative Commons Attribution-NonCommercial-ShareAlike 3.0" | https://zenodo.org/records/160761 | 44.1 kHz | 2 | 9 listener positions x 181 head orientations (-90..+90 deg, 1 deg) = 1629 measurements x 5 loudspeakers (5.0 surround); 44100 samples | Calypso listening room, TU Berlin, 83 m3, RT60 0.17 s at 1 kHz | KEMAR 45BA, large pinnae |
| conference_ilmenau (**HOLD, licence conflict**) | BRIRs-ConferenceRoom_5Positions.zip (5 SOFA confroom_pos1-5) | Zenodo field cc-by-4.0. BUT the SOFA files' own `License` attribute says "CC BY-NC-SA 4.0" | https://zenodo.org/records/163617 | 44.1 kHz | 2 | 5 listener positions, 72 azimuths (0-355 deg, 5 deg), 1 loudspeaker; 44100 samples | small conference room 10.3 x 5.8 x 3.1 m, RT 0.65 s | KEMAR 45BA, large ears |

Extra files in `data/brir/` that are NOT rows above (notes only): `surrey_cortex/LICENSE`, `surrey_cortex/BRIR_measurements.pdf` (Surrey measurement notes), `calypso/doc.zip` (Zenodo doc archive, same record).

The two HOLD rows: the record page (the official page) says CC BY 4.0, but the file itself says non-commercial. Treat as UNRESOLVED. Use for private training experiments only, ship nothing derived from them, until Johnny decides or the author confirms.

Not downloaded from the sets: Surrey 16 kHz copies (redundant), Calypso single-position files and the Genelec 8250A "Large" file (the combined file covers all nine positions).

## Beds (`data/beds/`)

Licence re-confirmed on each Freesound page (licence link on the page). Original lossless files need a Freesound login, so no login was used. The public HQ preview (MP3, about 128 kbps) was taken instead; all are 44.1 kHz or higher. Previews are lossy: fine for ambience beds, not for fine spectral work. Lossless originals: "needs Johnny download" if wanted (URLs below).

| File | Sound (uploader) | Licence | Page | Preview rate / ch / length | Original per page |
|---|---|---|---|---|---|
| 13747_28867-hq.mp3 | Scottish Restaurant.WAV (inchadney) | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) | https://freesound.org/people/inchadney/sounds/13747/ | 48 kHz, 2 ch, 240.6 s | 48 kHz, 16 bit, "Stereo" (the frontier note says binaural; page only says Stereo), 4:00 |
| 422097_1661766-hq.mp3 | Restaurant Atmosphere (crowded) (felix.blume) | CC0 (https://creativecommons.org/publicdomain/zero/1.0/) | https://freesound.org/people/felix.blume/sounds/422097/ | 48 kHz, 2 ch, 600.7 s | 96 kHz, 24 bit, Stereo, 10:00 (330 MB) |
| 252704_623488-hq.mp3 | 140812_Vienna_CafeZentral_R.wav (blaukreuz) | CC0 | https://freesound.org/people/blaukreuz/sounds/252704/ | 48 kHz, 2 ch, 100.4 s | 48 kHz, 24 bit, Stereo, 1:40 |
| 421647_8224400-hq.mp3 | Jazzy cafe ambience.wav (dersinnsspace) | CC0 | https://freesound.org/people/dersinnsspace/sounds/421647/ | 44.1 kHz, 2 ch, 84.8 s | 44.1 kHz, 32 bit, Stereo, 1:24 |
| 501025_37745-hq.mp3 | NYC diner ambiance (edlundart) | CC0 | https://freesound.org/people/edlundart/sounds/501025/ | 48 kHz, 1 ch, 315.0 s | 48 kHz, 32 bit, Mono, 5:15 |
| 424863_250867-hq.mp3 | French Cafe in the Morning (take 1) (alexanderweidner) | CC BY 4.0 | https://freesound.org/people/alexanderweidner/sounds/424863/ | 48 kHz, 2 ch, 600.0 s | 48 kHz, 24 bit, Stereo, 10:00 |
| 445094_8288383-hq.mp3 | Coffeehouse Ambience (mars_98) | CC BY 4.0 | https://freesound.org/people/mars_98/sounds/445094/ | 48 kHz, 2 ch, 120.4 s | 48 kHz, 24 bit, Stereo, 2:00 |
| 46310_43834-hq.mp3 | 0105coffeehouse.wav (moxobna) | CC0 | https://freesound.org/people/moxobna/sounds/46310/ | 44.1 kHz, 1 ch, 380.6 s | 44.1 kHz, 16 bit, Mono, 6:20 |

Needs Johnny download (optional, lossless): the 8 page URLs above, after logging in. Most useful: 422097 (96 kHz) and 252704, 424863, 445094 (24 bit).

## Rejected / not downloaded

| Item | Reason |
|---|---|
| Rostock Audio Lab BRIRs (depositonce 87.6) | Licence CC BY 4.0 confirmed on the item page, but `BRIRs.zip` is 14.1 GB; with the rest it would pass the 15 GB cap. Johnny decision. https://doi.org/10.14279/depositonce-87.6 |
| eBrIRD (UCL) | Page https://www.phon.ucl.ac.uk/resource/ebrird could not be reached (connection timeout from this machine). Licence UNVERIFIED, so skipped. |
| Aachen AIR | Licence UNVERIFIED (page not checked here). Skipped. |
| WDR Broadcast Studios | CC BY-SA 3.0. Other sets cover the need, so not downloaded. |
| BRAS (TU Berlin) | CC BY-SA 4.0 (per frontier.md). Not needed. |
| GRAS (RWTH Aachen) | CC BY-SA 4.0 (per ASH wiki). Not needed. |
| Salford BBC BRIR | CC BY-NC-SA 4.0 (per ASH wiki) |
| Seminar Room, TU Ilmenau (zenodo 206860) | CC BY-NC 4.0 (per ASH wiki) |
| Room HU201, TU Ilmenau (zenodo 61083) | CC BY-NC-SA 4.0 (per ASH wiki) |
| Rehabilitation Room, TU Ilmenau | CC BY-NC 4.0 (per ASH wiki) |
| LOCATA | Not downloaded by instruction (goes to Google Drive via Colab). Zenodo record 3630471, ODC Attribution per frontier.md. |
| Oldenburg cafeteria | Research/education only |
| WHAM! | CC BY-NC 4.0 |
| Speedenza binaural coffee shop, arnaud coutancier | CC BY-NC |
| elaski | Sampling+ |
| soundslikewillem, Anika11 | Rejected on licence in frontier.md; not re-checked |
| ARTE Zenodo cafes | Contradictory "non-commercial" note |

The ASH wiki list (https://github.com/ShanonPearce/ASH-Listening-Set/wiki/Datasets) was used only to find sets. Every accepted licence above was re-read on the set's own page or in its own files, except where marked "(per ASH wiki)" in the rejected list.

## Attribution (needed for CC BY items; MIT needs the licence text kept)

- Surrey Cortex rooms (MIT): keep `surrey_cortex/LICENSE`. Credit: "Binaural Room Impulse Responses Captured in Real Rooms, Copyright (c) 2016 Institute of Sound Recording, University of Surrey, MIT licence, https://github.com/IoSR-Surrey/RealRoomBRIRs".
- Spirit: "Binaural room impulse responses recorded with KEMAR in a small meeting room", H. Wierstorf and M. Geier, TU Berlin, Zenodo, 2016, https://doi.org/10.5281/zenodo.160751, CC BY 4.0. Indicate any changes.
- Auditorium 3: "Binaural room impulse responses recorded with KEMAR in a mid-size lecture hall", H. Wierstorf and M. Geier, TU Berlin, Zenodo, 2016, https://doi.org/10.5281/zenodo.160749, CC BY 4.0.
- ADREAM: "Binaural room impulse responses of an apartment-like environment", F. Winter, H. Wierstorf, A. Podlubne, T. Forgue, J. Manhes, M. Herrb, S. Spors, A. Raake, P. Danes, Zenodo, 2016, https://doi.org/10.5281/zenodo.49357, CC BY 4.0.
- Calypso (only if cleared): "Binaural room impulse responses of a 5.0 surround setup for different listening positions", H. Wierstorf, Zenodo, 2016, https://doi.org/10.5281/zenodo.160761, CC BY 4.0 per record; file metadata conflicts.
- Conference room (only if cleared): "Binaural room impulse responses: Same listener-source-setup at different positions in the room", A. Neidhardt, Zenodo, 2016, https://doi.org/10.5281/zenodo.163617, CC BY 4.0 per record; file metadata conflicts.
- "Scottish Restaurant.WAV" by inchadney, https://freesound.org/people/inchadney/sounds/13747/, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Indicate changes (MP3 preview, resampling, mixing).
- "French Cafe in the Morning (take 1)" by alexanderweidner, https://freesound.org/people/alexanderweidner/sounds/424863/, CC BY 4.0.
- "Coffeehouse Ambience" by mars_98, https://freesound.org/people/mars_98/sounds/445094/, CC BY 4.0.
- CC0 beds (felix.blume 422097, blaukreuz 252704, dersinnsspace 421647, edlundart 501025, moxobna 46310): no credit required; courtesy credit is fine.
