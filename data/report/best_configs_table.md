| Dataset | Mode | Primary | Best (μ±σ) | Variant | Flip | n | Baseline (μ±σ) | Δ | Effect | Note |
|---|---|---|---|---|---|---:|---|---:|---:|---|
| 20newsgroups | offline | accuracy | 0.3906 ± 0.0681 | backprop_ternary_step | ternary (per_step) | 3 | 0.1771 ± 0.0722 | 0.2135 | 3.043 |  |
| 20newsgroups | real | accuracy | 0.3206 ± 0.0046 | dfa_float_lr15 | off (off) | 3 | 0.3206 ± 0.0046 | 0.0000 | 0.000 |  |
| adult | offline | accuracy | 0.9089 ± 0.0316 | dfa_float_lr15 | off (off) | 3 | 0.9089 ± 0.0316 | 0.0000 | 0.000 |  |
| adult | real | accuracy | 0.8595 ± 0.0002 | dfa_float | off (off) | 3 | 0.8595 ± 0.0002 | 0.0000 | 0.000 |  |
| ag_news | offline | accuracy | 0.4813 ± 0.1840 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 | 0.4125 ± 0.0704 | 0.0688 | 0.493 |  |
| ag_news | real | accuracy | 0.9005 ± 0.0007 | backprop_float | off (off) | 2 | 0.9005 ± 0.0007 | 0.0000 | 0.000 |  |
| california_housing | offline | r2 | 0.0840 ± 0.0556 | dfa_float | off (off) | 3 | 0.0840 ± 0.0556 | 0.0000 | 0.000 |  |
| california_housing | real | r2 | 0.5561 ± 0.0065 | dfa_float | off (off) | 5 | 0.5561 ± 0.0065 | 0.0000 | 0.000 |  |
| cifar10 | offline | accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 | 1.0000 | 0.0000 | — | saturated |
| fashion_mnist | offline | accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 | 1.0000 | 0.0000 | — | saturated |
| fashion_mnist | real | accuracy | 0.8738 ± 0.0026 | backprop_float_lr15 | off (off) | 5 | 0.8738 ± 0.0026 | 0.0000 | 0.000 |  |
| mnist | offline | accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 | 1.0000 | 0.0000 | — | saturated |
| mnist | real | accuracy | 0.9646 ± 0.0003 | backprop_float_lr15 | off (off) | 5 | 0.9646 ± 0.0003 | 0.0000 | 0.000 |  |
| ucr | offline | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | — | saturated |
| ucr | real | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | — | saturated |
