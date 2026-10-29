# Blocked — needs Johnny (morning list, 2026-10-29)

- **Training speed (decision):** local training is off: the laptop had ~1 GB RAM free and the run swapped (GPU at 5 %), and your Phase 5 call was to keep training on Colab. Free T4 trains one model at a time (~8 h each, Phase 5 pace) and the queue is 5+ runs (swap, swap+mask running, all, Conformer, pretrained listener, then v3 retrains). Options: (a) free T4 only, slow; (b) spend compute units on an L4/A100 to cut that a lot. Your call.
- **Colab without popups:** every new Colab runtime asks for the Drive permission in a separate window the ship cannot click. Fix: save checkpoints/data to a private Hugging Face repo with a token you add once to Colab Secrets (the ship never handles the token).
- **Listening check (phase7-C6):** three 30 s sample v3 scenes will be in `data/samples_v3/` when the renderer lands.
- **Licence holds (optional):** Calypso and the Ilmenau conference room: Zenodo says CC BY, but the files themselves say non-commercial, so they are excluded. Rostock Audio Lab BRIRs are CC BY but 14 GB: download or skip? Lossless café originals (Freesound login) would beat the MP3 previews, mainly felix.blume 422097 (96 kHz).
- **Site review:** on hold until the new harness (Johnny 2026-10-28).
