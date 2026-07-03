Community Signal: DFA, Low‑Precision, and Measurement Rigor
===========================================================

- DFA is real, useful—and controversial at scale.
  - Feedback Alignment (FA) and Direct Feedback Alignment (DFA) reliably learn on small/medium MLPs and some non‑vision tasks, sometimes approaching backprop baselines with the right normalization/architecture.
  - Scaling to deep conv nets and ImageNet‑class tasks is sensitive; structured/learned feedback or architectural tweaks are often required.
  - Pointers: Lillicrap et al., Nature Communications (2016); Nøkland, ICLR (2016); Launay et al., NeurIPS/ICLR (2020); assorted arXiv follow‑ups.

- Evidence cuts both ways; scope claims and pick baselines carefully.
  - Positive: DFA variants can match BP on some large models or modalities when the backward path is adapted (e.g., orthonormal/Hadamard/learned feedback; normalization; residual structure).
  - Negative: FA/DFA degrade on deep conv stacks and ImageNet‑scale unless the backward path is adapted or co‑trained.
  - Pointers: Bartunov et al., NeurIPS (2018); Launay et al., (2020); OpenReview threads show architecture sensitivity and tuning needs.

- Low‑precision training best practices.
  - Keep float “shadow” weights; quantize only the forward path during activations.
  - Use stochastic rounding/dither to avoid bias and weight stagnation; keep determinism by seeding the RNG.
  - Pointers: BinaryConnect (Courbariaux et al., 2015), BNN (Hubara et al., 2016), DoReFa (Zhou et al., 2016), Ternary Weight Networks (Li et al., 2016), QSGD (Alistarh et al., 2017).

- Sign‑based convergence intuition.
  - If the update has a sign advantage over noise (p > 1/2), you can show descent or stationarity in noisy regimes; this underpins signSGD/QSGD‑style arguments.
  - Pointers: signSGD (Bernstein et al., 2018), QSGD (Alistarh et al., 2017), stochastic rounding analyses.

- Structured projections stabilize signals.
  - Orthonormal or Hadamard feedback approximately preserves norms and improves conditioning of the DFA signal (JL/FJLT intuition); alignment tends to improve with near‑isometric projections.
  - Pointers: Ailon & Chazelle, FJLT (2009); Launay et al., (2020); standard JL references.

- Reproducibility and statistics are table stakes.
  - NeurIPS/ICLR checklists expect explicit seeds, splits, hardware, software versions, code links, and error bars over multiple runs.
  - Prefer non‑parametric tests across datasets/settings (Demšar, 2006; Dror et al., 2018) when claiming method‑level differences.

- “Greener/edge‑ready” claims need numbers.
  - Use accepted yardsticks: MLPerf Tiny and EEMBC MLMark. Report accuracy–latency–energy trade‑offs, not accuracy alone. On ad‑hoc hardware, provide wall‑clock throughput and platform‑specific power sampling with clear methodology and caveats.

How this repo aligns
--------------------

- DFA scope: results are limited to small MLPs; we avoid ImageNet/large CNN claims. See README “Scope & claims”.
- Structured feedback: orthonormal/Hadamard options implemented; see docs/structured_feedback.md.
- Low precision: ternary forward path with float shadow weights; deterministic and stochastic (seeded) variants.
- Reproducibility: fixed seeds, offline data fixtures, repeat‑run summaries; see docs/reproducibility.md.
- Green claims: we do not claim energy/latency gains beyond throughput proxies. For measuring, see docs/energy_latency.md.

Suggested citations (non‑exhaustive)
------------------------------------

- Lillicrap et al., “Random synaptic feedback weights support error backpropagation for deep learning.” Nature Communications, 2016.
- Nøkland, “Direct Feedback Alignment Provides Learning in Deep Neural Networks.” ICLR, 2016.
- Bartunov et al., “Assessing the Scalability of Biologically‑Motivated Deep Learning Algorithms and Architectures.” NeurIPS, 2018.
- Launay, Poli, Krzakala, “Direct Feedback Alignment scales to modern deep learning tasks.” 2020.
- Courbariaux et al., “BinaryConnect.” NIPS, 2015. Hubara et al., “Binarized Neural Networks.” NIPS, 2016. Zhou et al., “DoReFa‑Net.” arXiv, 2016. Li et al., “Ternary Weight Networks.” NIPS, 2016.
- Alistarh et al., “QSGD: Communication‑Efficient SGD via Gradient Quantization and Encoding.” NeurIPS, 2017. Bernstein et al., “signSGD.” ICML/NeurIPS, 2018.
- Ailon & Chazelle, “The Fast Johnson–Lindenstrauss Transform.” SICOMP, 2009.
- Demšar, “Statistical Comparisons of Classifiers over Multiple Data Sets.” JMLR, 2006. Dror et al., “The Hitchhiker’s Guide to Statistical Significance Tests for NLP.” 2018.

