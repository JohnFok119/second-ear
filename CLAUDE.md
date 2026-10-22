# second-ear

Sound-localization model + one-ear experiment + 3D coffee-shop viewer. Hands-off run; Johnny checks milestone pages.

- Design: `ARCHITECTURE.md`. Chunks: `PLAN.md` (tick, commit, merge to main, clear, next). Facts + URLs: `docs/research/`.
- Stack: Python 3.11 via `uv` (`.venv`), PyTorch CUDA, pytest, ruff. Viewer: `web/`, plain HTML + Three.js (pinned CDN), no build step.
- Data in `data/` (git-ignored). Results JSON in `results/` (tracked). Checkpoints in `checkpoints/` (ignored).
- Run `uv run pytest` before any commit. Never run `python3`.
- Standing go (Johnny, 2026-10-22): merge feature branches to `main` and dispatch crew without asking, this repo only. Cloud spend $0.
- Spine: `personal\private\projects\second-ear\` (context, decisions, logs). Resume via "continue".

<!-- fleet:autonomy:start -->
## Autonomy rules (fleet)

- Work only on a branch whose name starts with `claude/`. Never work on or push to `main` or `master`.
- Never read or print secrets or `.env` files.
- Before every push, run this repo's verify command (tests and/or build). Push only if it passes.
- Say "done" only when the task's written stop condition is checkably met. Show the check.
- If a fact is missing, write the question into `BLOCKED.md` at the repo root and move on to other work. Do not guess.
- This repo may lack GitHub branch protection; a push-guard hook blocks pushes to main/master, force-pushes and deletes. Never try to work around it.
- Stay inside this repo. Do not touch anything outside it.
<!-- fleet:autonomy:end -->
