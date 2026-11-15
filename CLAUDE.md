# second-ear

Sound-localization model + one-ear experiment + 3D coffee-shop viewer. Hands-off run; Johnny checks milestone pages.

- Design: `ARCHITECTURE.md`. Chunks: `PLAN.md` (tick, commit, merge to main, clear, next). Facts + URLs: `docs/research/`.
- Stack: Python 3.11 via `uv` (`.venv`), PyTorch CUDA, pytest, ruff. Viewer: `web/`, plain HTML + Three.js (pinned CDN), no build step.
- Data in `data/` (git-ignored). Results JSON in `results/` (tracked). Checkpoints in `checkpoints/` (ignored).
- Run `uv run pytest` before any commit. Never run `python3`.
