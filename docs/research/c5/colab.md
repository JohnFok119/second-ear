# Phase 5 research: Google Colab for training (2026-10-24, crew-researcher)

- Colab Pro for Education (free year for US students) is closed to new sign-ups; verified users keep it to their 1-year mark. https://research.google.com/colaboratory/faq.html
- Google AI Pro free year for US college students (sign-up to 2026-12-31) does not mention Colab (source: MacRumors 2026-08-19; Google page not readable, UNVERIFIED). https://macrumors.com/2026/08/19/students-free-year-google-ai-pro
- Free tier: sessions at most 12 h, limits vary and are unpublished; GPU usually T4 (UNVERIFIED which). Disk ~100 GB per a third-party guide (UNVERIFIED).
- Data: keep Drive folders under ~10k items; copy one zip from Drive to the VM disk and unzip there.
- GPU FP32 vs local Quadro RTX 4000 Max-Q (8 GB, ~7.1 TFLOPS): T4 16 GB ~1.1×, A100 ~2.8× (far more with tensor cores), L4 24 GB ~4.3×. https://www.nvidia.com/en-us/data-center/tesla-t4/ · https://www.nvidia.com/en-us/data-center/l4/
- Bottom line: Colab helps only for a model that needs more than 8 GB or a faster paid GPU; for our small CRNN the free T4 is no faster than the laptop and sessions can be cut. Check Runtime → Change runtime type for L4/A100 to see whether the account has a paid tier.
