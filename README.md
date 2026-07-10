FeedFlipNets
===========

[![DOI](https://zenodo.org/badge/1017107032.svg)](https://doi.org/10.5281/zenodo.21152011)

FeedFlipNets trains ternary networks by **feedback-driven bit-flips**: weights are stored only as ternary `{-1,0,+1}` plus a small integer flip-accumulator (**no float shadow weights**), and a cheap feedback signal flips bits directly. A flip needs only the *sign* of "should this weight go up or down," so Direct Feedback Alignment (DFA) — a transport-free, shadow-free, lock-free signal — is the natural feedback. The project introduces **Activation-Routed DFA (AR-DFA)** to supply a better transport-free sign, and establishes the central finding that **per-weight sign correctness — not cosine alignment — is the binding constraint** for transport-free discrete/ternary learning.

- Paper: [`docs/paper/main.pdf`](docs/paper/main.pdf)
- Findings + numbers: [`data/report/`](data/report/) (per milestone: `m1`, `m2`, `m2b`, `speed`, `feedflip`)
- Cite: [`CITATION.cff`](CITATION.cff) · DOI [10.5281/zenodo.21152011](https://doi.org/10.5281/zenodo.21152011) (all versions)

Status: a research artifact accompanying the paper. It is **not** a backprop replacement and **not** a mature training library — see *Scope & honest caveats* below.


What it is
----------

Three contributions, each pre-registered with numeric gates and honestly recorded (including the negatives):

1. **The FeedFlip bit-flip mechanism.** Shadow-free, sign-driven ternary training: `c += -sign(grad)`; when the accumulator `|c|` crosses a threshold, flip the bit and reset. With an *accurate* sign it matches float-shadow backpropagation at **~5.6× less optimizer state and ~2× faster steps** on an MLP.

2. **Activation-Routed DFA (AR-DFA).** A transport-free feedback rule that reuses the parts of a Transformer block's Jacobian already cached in the forward pass — the softmax routing matrix `A`, its Jacobian, the LayerNorm scale, the nonlinearity mask — and surrogates *only* the weight-transposes. This makes the value-path gradients **exact** (matching autograd to ~1e-15) and, with a perturbation-taught surrogate, cuts the worst-case attention-block **cosine**-alignment angle from ~90° to **46°**, all lock-free (verified by a probe with a positive control).

3. **The per-weight sign barrier (the central, negative result).** AR-DFA's alignment gain does *not* help bit-flipping: on a ternary Transformer, transport-free flipping is worse than not flipping, and AR-DFA does not beat vanilla DFA — because per-weight **sign**-match to the true gradient stays near chance (`p ≈ 0.53`) *even as* cosine alignment improves sharply. **Aggregate (cosine) alignment is the wrong proxy; per-weight sign correctness is the binding constraint.** Corollaries (in `data/report/`): DFA is not backprop-competitive on accuracy, per-step cost, or wall-clock-to-target.


Key results
-----------

| finding | numbers | source |
| --- | --- | --- |
| Bit-flips match float-shadow BP with a good sign | 0.629 vs 0.619 acc, **5.6× less state** | `data/report/feedflip/` |
| AR-DFA value-path gradients are exact | `≤ 3.1e-15` vs autograd | `data/report/m2/` |
| AR-DFA cuts worst-case attention alignment | 90.6° → **46.4°** (transport-free) | `data/report/m2/` |
| …but per-weight sign-match stays ~chance | `p ≈ 0.53` (fixed-DFA and AR-DFA alike) | `data/report/feedflip/` |
| Transport-free flipping fails on a Transformer | worse than freezing the blocks | `data/report/feedflip/` |
| 3 pre-registered attack rounds on the gap: all NO-GO | best ≤8 bits/w arm 0.574 vs shadow 0.604 | `data/report/feedflip/` |
| The gap is a *state-bits frontier*, not sign fidelity | p=1.0 signs → 0.551; +magnitude votes → 0.574 | `data/report/feedflip/` |
| AR-DFA is not backprop-competitive (char-LM) | +1.93 bpc, ~30× slower/step | `data/report/m2b/` |


Install
-------

```bash
git clone https://github.com/Akhilesh-Gogikar/FeedFlipNets.git
cd FeedFlipNets
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip && pip install -e .
```

Requirements: Python 3.8+. The AR-DFA / bit-flip Transformer experiments use CPU-only PyTorch (`requirements-extras.txt`). CPU is sufficient throughout — no GPU required.


Reproduce the findings
----------------------

```bash
# Full test suite (grad-checks, lock-free probe + positive control, alignment metrics)
pytest -q

# Analytical experiments (self-contained)
python -m experiments.pipeline_cost_model            # DFA-pipeline vs tuned 1F1B throughput model
python -m experiments.accuracy_transport_frontier    # how much transport buys back accuracy
```

- The **mechanism + guards** live in `feedflipnets/core/` (`deep_mlp`, `transformer`, `lm`, `strategies`) and `feedflipnets/eval/` (`gradcheck`, `alignment`, `lockfree`).
- Every result table has a committed JSON + README under `data/report/{m1,m2,m2b,speed,feedflip}/`.
- Design specs and pre-registered plans are under `docs/superpowers/{specs,plans}/`.


Build the paper
---------------

```bash
cd docs/paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
# or, if installed: latexmk -pdf main.tex   /   tectonic main.tex
open docs/paper/main.pdf   # macOS
```

The source defaults to a blinded build (author "Anonymous"); set `\blindedfalse` for a camera-ready (non-anonymous) PDF.


Original ternary-DFA baseline harness (legacy)
----------------------------------------------

The repository also retains the earlier deterministic ternary-DFA baseline stack — float "shadow" weights `V`, ternary forward `W = Q_τ(V)`, DFA updates, structured (orthogonal/Hadamard) feedback — with CLI presets and offline fixtures. This is orthogonal to the contributions above and summarized in the paper's appendix.

```bash
FEEDFLIP_DATA_OFFLINE=1 pytest -q tests/test_datasets_smoke.py tests/test_training_loops.py
python -m cli.main --preset mnist_mlp_dfa --feedback dfa --flip ternary --flip-schedule per_step
make report   # aggregate legacy sweeps into data/report/
```


Repository layout
-----------------

- `feedflipnets/core/` — deep MLP, Transformer block, stackable LM, feedback strategies (incl. AR-DFA)
- `feedflipnets/eval/` — grad-check, alignment probe, lock-free probe (+ positive control)
- `experiments/` — analytical cost model and transport-frontier scripts
- `data/report/{m1,m2,m2b,speed,feedflip}/` — per-milestone results (JSON + README)
- `docs/paper/` — LaTeX manuscript · `docs/superpowers/` — specs and plans
- `cli/`, `configs/presets/`, `datasets/`, `scripts/` — legacy ternary-DFA baseline harness


Scope & honest caveats
----------------------

- **Not a backprop replacement.** Across accuracy, per-step cost, wall-clock-to-target, and bit-flip quality, DFA (including AR-DFA) does not beat backpropagation; the value is the efficient bit-flip *mechanism* and the *characterization* of why transport-free learning is hard.
- **Small-scale, CPU, reproducibility-first.** Single-head blocks, small corpora, modest depth — chosen so every claim is exactly reproducible and reference gradients are affordable. The positive claims are bounded by this scale; the negative results are corroborated across independent axes.
- **A research artifact, not a library.** No stable public API is promised.


Citation & License
------------------

License: MIT (see [`LICENSE`](LICENSE)). Cite via [`CITATION.cff`](CITATION.cff) or:

```bibtex
@software{gogikar_feedflipnets_2026,
  title     = {FeedFlipNets: Feedback-Driven Bit-Flips for Ternary Networks, Activation-Routed DFA, and the Per-Weight Sign Barrier to Transport-Free Learning},
  author    = {Gogikar, Akhilesh},
  year      = {2026},
  version   = {2.0.0-rc1},
  doi       = {10.5281/zenodo.21152011},
  url       = {https://github.com/Akhilesh-Gogikar/FeedFlipNets}
}
```

Keywords: ternary networks · bit-flip training · direct feedback alignment · weight transport · sign correctness · attention · reproducible research.
