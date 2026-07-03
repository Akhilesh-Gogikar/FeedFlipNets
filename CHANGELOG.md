# Changelog

## [2.0.0-rc1] - 2026-07-02
### Added
- **FeedFlip** bit-flip training rule for ternary networks: shadow-free (ternary
  weight + small integer flip-accumulator), sign-driven updates. Matches
  float-shadow BP on an MLP at ~5.6x less optimizer state and ~2x faster steps
  when given an accurate sign. Study in `data/report/feedflip/`.
- **Activation-Routed DFA (AR-DFA)**: a transport-free feedback rule that reuses a
  Transformer block's cached data-maps (softmax routing, its Jacobian, LayerNorm
  scale, nonlinearity mask) and surrogates only the weight transposes. Yields
  value-path-exact gradients (`feedflipnets/core/{transformer,lm}.py`,
  `strategies.py`) and cuts worst-case attention cosine-alignment from ~90 to ~46
  degrees, all lock-free (probe with positive control in `feedflipnets/eval/`).
- Pre-registered milestones M1/M2/M2b with numeric gates and honestly-recorded
  NO-GOs (`docs/superpowers/{specs,plans}/`, `data/report/{m1,m2,m2b}/`), a
  pipeline cost model and transport-fraction frontier (`data/report/speed/`,
  `experiments/`), and the alignment-measurement toolkit
  (grad-check, lock-free probe, per-matrix cosine/theta/depth-slope).
- Reworked paper (`docs/paper/main.pdf`) centered on the per-weight **sign
  barrier**: cosine alignment (which AR-DFA improves) is the wrong proxy;
  per-weight sign correctness is the binding constraint for transport-free
  discrete/ternary learning.

### Findings (honest negatives)
- Transport-free bit-flipping fails on a ternary Transformer (worse than freezing
  the blocks); AR-DFA does not rescue it (per-weight sign-match ~0.53, ~chance).
- AR-DFA's alignment gains do not translate into competitive training (accuracy,
  per-step cost, or wall-clock-to-target). DFA is not a backprop replacement.

## [1.0.0-rc1] - 2024-06-12
### Added
- Offline-first presets for MNIST, UCR GunPoint, California Housing, and
  20 Newsgroups under `configs/presets/`, plus a configurable sweep helper in
  `scripts/preset_sweep.py` for grid-searching feedback, flip modes and schedules,
  learning rates, and hidden sizes.
- GitHub Actions workflow covering Python 3.10/3.11 linting, formatting, tests,
  and preset smoke runs that upload metrics artifacts.
- Pre-commit configuration (`.pre-commit-config.yaml`), extended Makefile
  targets (`setup`, `format`, `lint`, `test`, `smoke`, `run`), and repository
  documentation (`README.md`, `CONTRIBUTING.md`, `docs/how_to_add_dataset.md`).
- Deterministic offline dataset fixtures for MNIST, UCR time series, and
  20 Newsgroups that achieve above-random accuracy during smoke runs.

### Changed
- `feedflipnets/training/pipelines.py` now discovers presets from
  `configs/presets/`, validates optimiser selections, and exposes them via the
  CLI.
- Dependency metadata updated with PyYAML, linting, and tooling packages to
  support the new presets and developer workflow.

### Fixed
- Offline dataset generators produce structured signals so loss decreases and
  accuracy exceeds random chance in regression and classification presets.
