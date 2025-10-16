NeurIPS‑Style Reproducibility Checklist (FeedFlipNets)
======================================================

Paper clarity
-------------

- [x] Problem setting and contributions are clearly stated.
- [x] Claims are scoped (small MLPs; no large‑scale CNN claims).
- [x] Limitations and negative results are acknowledged.

Method details
--------------

- [x] Model architecture(s) fully specified (layer sizes, activations).
- [x] Training details provided (optimizer, LR, batch size, epochs).
- [x] Feedback strategies described (BP, DFA, structured, ternary DFA).
- [x] Quantization details given (threshold τ, schedules, det/stoch).

Datasets
--------

- [x] Datasets named and cited (MNIST, Fashion‑MNIST, 20NG, AG News, UCR GunPoint, California Housing).
- [x] Splits detailed and deterministic; validation selection procedure stated.
- [x] Dataset licenses are compatible with research use.

Evaluation
----------

- [x] Metrics defined (accuracy, R²; throughput and sparsity as auxiliaries).
- [x] Number of runs per configuration reported; error bars shown.
- [x] Baselines appropriate (BP, DFA float, structured DFA).
- [x] Hyperparameter sweeps described (LR, τ) and ranges justified.
- [x] Statistical tests recommended for aggregate claims (Demšar/Dror).

Compute and environment
-----------------------

- [x] Hardware specified (template provided; fill exact CPU/RAM).
- [x] Software versions pinned (`requirements-lock.txt`).
- [x] Random seeds fixed for all experiments.

Code, data, and artifacts
-------------------------

- [x] Code available with instructions (README Quickstart, presets).
- [x] Exact configs saved next to metrics for every run.
- [x] Scripts to reproduce main tables/figures (`reproduce_all.sh`, `make report`, `make paper`).
- [x] Artifacts (CSV/JSON/plots) included or auto‑generated.

Safety, ethics, and broader impacts
-----------------------------------

- [x] No sensitive data; tasks are standard benchmarks.
- [x] Potential misuse is minimal; results are baseline‑oriented.
- [x] Broader impacts discussed at a high level (deployability vs. accuracy trade‑offs).

Notes
-----

- A formal Reproducibility Statement is provided in `docs/ReproducibilityStatement.md`.

