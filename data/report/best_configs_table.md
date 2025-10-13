| Dataset | Mode | Primary | Best (μ±σ) | Variant | Flip | n | Baseline (μ±σ) | Δ | Effect | Note |
|---|---|---|---|---|---|---:|---|---:|---:|---|
| 20newsgroups | offline | accuracy | 0.3906 ± 0.0681 | backprop_ternary_step | ternary (per_step) | 3 | 0.1771 ± 0.0722 | 0.2135 | 3.043 |  |
| 20newsgroups | real | accuracy | 0.3206 ± 0.0046 | dfa_float_lr15 | off (off) | 3 | 0.3206 ± 0.0046 | 0.0000 | 0.000 |  |
| california_housing | offline | r2 | 0.0840 ± 0.0556 | dfa_float | off (off) | 3 | 0.0840 ± 0.0556 | 0.0000 | 0.000 |  |
| california_housing | real | r2 | nan ± nan | dfa_float_lr15 | off (off) | 3 | — | — | — |  |
| mnist | offline | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | — | saturated |
| mnist | real | accuracy | 0.9649 ± 0.0001 | backprop_float_lr15 | off (off) | 3 | 0.9649 ± 0.0001 | 0.0000 | 0.000 |  |
| ucr | offline | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | — | saturated |
| ucr | real | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | — | saturated |
