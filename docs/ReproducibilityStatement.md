Reproducibility Statement (FeedFlipNets)
=======================================

Summary
-------

- Scope: deterministic DFA and ternary‑forward baselines on small, standard datasets using shallow MLPs. No large‑scale or SOTA claims.
- Artifacts: each run writes metrics/manifests under `runs/`; reports aggregate to `data/report/`.

Datasets and splits
-------------------

- Datasets: MNIST, Fashion‑MNIST subsets, 20 Newsgroups, AG News (small variant), UCR GunPoint, California Housing.
- Splits: fixed training/validation/test splits as provided by the loader; when applicable, we use canonical train/test from the source datasets and reserve a validation subset from train via a fixed seed.
- Licensing: each dataset is publicly available under its respective license; no restricted or proprietary data.

Training and evaluation
-----------------------

- Seeds: all experiments use fixed seeds for initialization and data shuffling; default seeds are recorded in each run’s manifest.
- Number of runs: per‑configuration repeats n in [2, 5]. We report mean ± standard deviation across repeats.
- Hyperparameters: learning rate, batch size, flip threshold τ, flip schedule (per_step/per_epoch), and quantization mode (det/stoch). Full configs are stored alongside metrics.
- Metrics: classification accuracy for classification tasks; R² for regression (California Housing). Throughput (samples/s) and sparsity (zero ratio) are reported as auxiliary metrics.
- Early stopping: not used; fixed‑epoch training for comparability across methods.

Hardware and software
---------------------

- Hardware: CPU‑only runs on a single machine. Please fill in exact details before submission: CPU model, core count, RAM, storage type.
- Software: Python 3.8+; see `requirements-lock.txt` for pinned versions. OS details and Python environment hash are captured in run manifests.

Compute budget and runtimes
---------------------------

- Typical runtimes (representative, to be updated with your machine):
  - MNIST‑MLP, 5 epochs, batch 32: O(minutes).
  - 20NG‑BoW‑MLP, 5 epochs, batch 64: O(minutes–tens of minutes).
  - UCR GunPoint‑MLP, 20 epochs: O(minutes).
- All experiments fit on a laptop‑class CPU with <16 GB RAM.

Code and artifacts
------------------

- Source code: this repository.
- Reproduction entrypoints: `./reproduce_all.sh`, `make bench`, and `python -m cli.main` with presets under `configs/presets/`.
- Artifacts: per‑run under `runs/`; aggregated under `data/report/` (CSV/JSON/plots). Paper build under `docs/paper/` (auto‑generates a Key Metrics table from CSVs).

Statistical reporting
---------------------

- Report mean ± standard deviation over n seeds; show per‑seed points where space allows.
- For multi‑dataset method claims, prefer non‑parametric tests across datasets/settings (e.g., Demšar/Dror protocols).

Ethics, limitations, and negative results
-----------------------------------------

- Limitations: DFA/FA stability and accuracy degrade on deep CNNs and large‑scale vision unless the backward path is adapted. Our results are scoped accordingly.
- Negative results: some structured feedback variants underperform on trivial tasks where BP saturates; we include such observations in plots/tables rather than hide them.
- Broader impacts: no sensitive data is used; models are small and educational/baseline‑oriented.

Checklist tick‑off (for submission forms)
-----------------------------------------

- [x] Datasets named and licensed
- [x] Splits defined and seeds fixed
- [x] Hardware/software specified or template provided
- [x] Code and configs available
- [x] Number of runs and error bars stated
- [x] Evaluation metrics and protocols described
- [x] Limitations and negative results acknowledged

