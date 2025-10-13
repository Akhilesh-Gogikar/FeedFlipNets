# FeedFlipNets Benchmark Summary

Aggregated over seeds with mean ± std.


## Topline Highlights

| Dataset | Mode | Metric | Mean ± Std | Strategy Variant | Flip | n |
|---|---|---|---|---|---|---:|
| 20newsgroups | offline | Accuracy | 0.3906 ± 0.0681 | backprop_ternary_step | ternary (per_step) | 3 |
| 20newsgroups | offline | Macro-F1 | 0.3414 ± 0.0474 | backprop_ternary_step | ternary (per_step) | 3 |
| 20newsgroups | offline | Zero Ratio | 0.6819 ± 0.0001 | structured_orth_ternary | ternary (per_step) | 3 |
| 20newsgroups | offline | Test Throughput (samples/s) | 17760.0556 ± 558.8727 | backprop_float | off (off) | 3 |
| 20newsgroups | real | Accuracy | 0.3206 ± 0.0046 | dfa_float_lr15 | off (off) | 3 |
| 20newsgroups | real | Macro-F1 | 0.2744 ± 0.0045 | dfa_float_lr15 | off (off) | 3 |
| 20newsgroups | real | Zero Ratio | 0.9544 ± 0.0000 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| 20newsgroups | real | Test Throughput (samples/s) | 17155.5581 ± 224.3879 | backprop_float | off (off) | 3 |
| adult | offline | Accuracy | 0.6438 ± 0.0892 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| adult | offline | R² | -0.0631 ± 0.0042 | backprop_float | off (off) | 5 |
| adult | offline | Zero Ratio | 0.6707 ± 0.0072 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| adult | offline | Test Throughput (samples/s) | 50088.4307 ± 6370.6315 | backprop_float | off (off) | 5 |
| adult | real | Accuracy | 0.8595 ± 0.0002 | dfa_float | off (off) | 3 |
| adult | real | R² | 0.4654 ± 0.0014 | dfa_float | off (off) | 3 |
| adult | real | Zero Ratio | 0.4453 ± 0.0051 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 3 |
| adult | real | Test Throughput (samples/s) | 41361.0390 ± 856.9266 | backprop_float | off (off) | 3 |
| ag_news | offline | Accuracy | 0.4813 ± 0.1840 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Macro-F1 | 0.3933 ± 0.2158 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Zero Ratio | 0.6818 ± 0.0003 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Test Throughput (samples/s) | 4356.2552 ± 945.0397 | backprop_float | off (off) | 5 |
| ag_news | real | Accuracy | 0.9005 ± 0.0007 | backprop_float | off (off) | 2 |
| ag_news | real | Macro-F1 | 0.9002 ± 0.0007 | backprop_float | off (off) | 2 |
| ag_news | real | Zero Ratio | 0.6721 ± 0.0008 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 2 |
| ag_news | real | Test Throughput (samples/s) | 3807.2768 ± 243.6141 | backprop_float | off (off) | 2 |
| california_housing | offline | R² | 0.0840 ± 0.0556 | dfa_float | off (off) | 3 |
| california_housing | offline | Zero Ratio | 0.2186 ± 0.2010 | structured_hadamard_ternary | ternary (per_step) | 3 |
| california_housing | offline | Test Throughput (samples/s) | 87757.7091 ± 1022.0328 | backprop_float | off (off) | 3 |
| california_housing | real | R² | nan ± nan | dfa_float_lr15 | off (off) | 3 |
| california_housing | real | Zero Ratio | 0.5957 ± 0.3504 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| california_housing | real | Test Throughput (samples/s) | 161174.3075 ± 1014.8411 | structured_hadamard_float | off (off) | 3 |
| fashion_mnist | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| fashion_mnist | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| fashion_mnist | offline | Zero Ratio | 0.9533 ± 0.0005 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| fashion_mnist | offline | Test Throughput (samples/s) | 3746.2850 ± 162.4724 | dfa_float_clip1 | off (off) | 5 |
| fashion_mnist | real | Accuracy | 0.8703 ± 0.0018 | backprop_float | off (off) | 5 |
| fashion_mnist | real | Macro-F1 | 0.8692 ± 0.0018 | backprop_float | off (off) | 5 |
| fashion_mnist | real | Zero Ratio | 0.6709 | structured_orth_ternary | ternary (per_step) | 1 |
| fashion_mnist | real | Test Throughput (samples/s) | 5519.9567 ± 178.3732 | structured_orth_float | off (off) | 5 |
| mnist | offline | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| mnist | offline | Macro-F1 | 1.0000 ± 0.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| mnist | offline | Zero Ratio | 0.6787 ± 0.0004 | structured_orth_ternary | ternary (per_step) | 3 |
| mnist | offline | Test Throughput (samples/s) | 53421.9733 ± 2514.3412 | backprop_float | off (off) | 3 |
| mnist | real | Accuracy | 0.9649 ± 0.0001 | backprop_float_lr15 | off (off) | 3 |
| mnist | real | Macro-F1 | 0.9647 ± 0.0001 | backprop_float_lr15 | off (off) | 3 |
| mnist | real | Zero Ratio | 0.7472 ± 0.0051 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| mnist | real | Test Throughput (samples/s) | 65656.5516 ± 1282.5006 | structured_hadamard_float | off (off) | 3 |
| ucr | offline | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | offline | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | offline | Zero Ratio | 0.6772 ± 0.0021 | structured_hadamard_ternary | ternary (per_step) | 3 |
| ucr | offline | Test Throughput (samples/s) | 51612.2241 ± 3049.0575 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Zero Ratio | 0.6772 ± 0.0021 | structured_hadamard_ternary | ternary (per_step) | 3 |
| ucr | real | Test Throughput (samples/s) | 41204.3096 ± 1394.5144 | structured_orth_float | off (off) | 3 |

## Best Configs

| Dataset | Mode | Primary | Best (μ±95% CI) | Variant | Flip | n | Baseline (μ±σ) | Δ | Effect Size |
|---|---|---|---|---|---|---:|---|---:|---:|
| 20newsgroups | offline | accuracy | 0.3906 ± 0.0771 | backprop_ternary_step | ternary (per_step) | 3 | 0.1771 ± 0.0722 | 0.2135 | 3.043 |
| 20newsgroups | real | accuracy | 0.3206 ± 0.0052 | dfa_float_lr15 | off (off) | 3 | 0.3206 ± 0.0046 | 0.0000 | 0.000 |
| adult | offline | accuracy | 0.6438 ± 0.0782 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 | 0.5422 ± 0.1061 | 0.1016 | 1.037 |
| adult | real | accuracy | 0.8595 ± 0.0003 | dfa_float | off (off) | 3 | 0.8595 ± 0.0002 | 0.0000 | 0.000 |
| ag_news | offline | accuracy | 0.4813 ± 0.1613 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 | 0.4125 ± 0.0704 | 0.0688 | 0.493 |
| ag_news | real | accuracy | 0.9005 ± 0.0009 | backprop_float | off (off) | 2 | 0.9005 ± 0.0007 | 0.0000 | 0.000 |
| california_housing | offline | r2 | 0.0840 ± 0.0629 | dfa_float | off (off) | 3 | 0.0840 ± 0.0556 | 0.0000 | 0.000 |
| california_housing | real | r2 | nan ± nan | dfa_float_lr15 | off (off) | 3 | nan ± nan | nan | 0.000 |
| fashion_mnist | offline | accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 | 1.0000 | 0.0000 | 0.000 |
| fashion_mnist | real | accuracy | 0.8703 ± 0.0016 | backprop_float | off (off) | 5 | 0.8703 ± 0.0018 | 0.0000 | 0.000 |
| mnist | offline | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | 0.000 |
| mnist | real | accuracy | 0.9649 ± 0.0001 | backprop_float_lr15 | off (off) | 3 | 0.9649 ± 0.0001 | 0.0000 | 0.000 |
| ucr | offline | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | 0.000 |
| ucr | real | accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 | 1.0000 | 0.0000 | 0.000 |

## 20newsgroups (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.1771 ± 0.0722 | 3 |
| backprop_float | off (off) | loss | 2.0570 ± 0.0084 | 3 |
| backprop_float | off (off) | macro_f1 | 0.1426 ± 0.0929 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 17760.0556 ± 558.8727 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 4783.4421 ± 118.0022 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.3906 ± 0.0681 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 12.0424 ± 1.1263 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3414 ± 0.0474 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6810 ± 0.0002 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 7772.4028 ± 238.0358 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 1040.2072 ± 21.3936 | 3 |
| dfa_float | off (off) | accuracy | 0.1510 ± 0.0325 | 3 |
| dfa_float | off (off) | loss | 2.0691 ± 0.0078 | 3 |
| dfa_float | off (off) | macro_f1 | 0.0780 ± 0.0426 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 7518.0989 ± 822.4189 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 2244.2749 ± 4.4090 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.3438 ± 0.0413 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 13.0543 ± 1.3608 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.2602 ± 0.0542 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6818 ± 0.0001 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 7491.7946 ± 264.3764 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 2256.5135 ± 75.8614 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.2240 ± 0.0502 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 15.7157 ± 1.0053 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.1734 ± 0.0701 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0001 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 7340.1342 ± 880.5489 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 1048.3695 ± 24.8185 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.1354 ± 0.0361 | 3 |
| structured_hadamard_float | off (off) | loss | 2.0756 ± 0.0078 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0825 ± 0.0410 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 17159.8569 ± 550.9908 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 3804.6909 ± 65.6131 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.2917 ± 0.0180 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 14.5478 ± 0.6010 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2593 ± 0.0096 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0001 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 7245.0925 ± 290.8769 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 975.0769 ± 13.9088 | 3 |
| structured_orth_float | off (off) | accuracy | 0.1354 ± 0.0361 | 3 |
| structured_orth_float | off (off) | loss | 2.0751 ± 0.0073 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.0817 ± 0.0410 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 17386.8935 ± 183.4235 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 1490.1215 ± 87.3035 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.2083 ± 0.0861 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 16.1435 ± 2.0061 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.1511 ± 0.0790 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0001 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 7804.6346 ± 167.3541 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 713.6274 ± 3.6794 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.2656 ± 0.0413 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 14.8916 ± 0.8861 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.2078 ± 0.0311 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6809 ± 0.0003 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 7573.5650 ± 385.2984 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 1054.6434 ± 42.0011 | 3 |

## 20newsgroups (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.1230 ± 0.0042 | 3 |
| backprop_float | off (off) | loss | 2.9214 ± 0.0026 | 3 |
| backprop_float | off (off) | macro_f1 | 0.0722 ± 0.0035 | 3 |
| backprop_float | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 17155.5581 ± 224.3879 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 4127.2925 ± 9.0704 | 3 |
| backprop_float_lr06 | off (off) | accuracy | 0.0862 ± 0.0165 | 3 |
| backprop_float_lr06 | off (off) | loss | 2.9690 ± 0.0004 | 3 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.0439 ± 0.0159 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 8381.7305 ± 415.6071 | 3 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 2063.9421 ± 85.7122 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.1230 ± 0.0042 | 3 |
| backprop_float_lr10 | off (off) | loss | 2.9214 ± 0.0026 | 3 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.0722 ± 0.0035 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 8209.2622 ± 596.1825 | 3 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 1774.0311 ± 384.0014 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.1986 ± 0.0134 | 3 |
| backprop_float_lr15 | off (off) | loss | 2.6976 ± 0.0102 | 3 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.1400 ± 0.0198 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 7069.3706 ± 1939.4838 | 3 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 1727.3163 ± 298.0806 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.0615 ± 0.0122 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 17.8996 ± 0.1939 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.0541 ± 0.0109 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6822 ± 0.0001 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 6125.3234 ± 104.1322 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 810.4823 ± 2.6101 | 3 |
| dfa_float | off (off) | accuracy | 0.2229 ± 0.0107 | 3 |
| dfa_float | off (off) | loss | 2.5541 ± 0.0192 | 3 |
| dfa_float | off (off) | macro_f1 | 0.1686 ± 0.0084 | 3 |
| dfa_float | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 6134.0114 ± 112.5139 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 1879.1527 ± 6.0185 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.1294 ± 0.0168 | 3 |
| dfa_float_lr06 | off (off) | loss | 2.8922 ± 0.0220 | 3 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.0769 ± 0.0103 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 1397.4045 ± 307.6902 | 3 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 489.7546 ± 124.7276 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.2229 ± 0.0107 | 3 |
| dfa_float_lr10 | off (off) | loss | 2.5541 ± 0.0192 | 3 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.1686 ± 0.0084 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 2329.9449 ± 458.5847 | 3 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 805.4994 ± 143.5994 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.3206 ± 0.0046 **(best)** | 3 |
| dfa_float_lr15 | off (off) | loss | 2.2206 ± 0.0104 | 3 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.2744 ± 0.0045 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 2433.5734 ± 653.8873 | 3 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 761.1861 ± 183.7896 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.1264 ± 0.0184 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 17.2000 ± 0.7207 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.0992 ± 0.0296 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6793 ± 0.0003 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 5723.3596 ± 202.3799 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 1843.8629 ± 18.0854 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.1028 ± 0.0143 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 18.3765 ± 0.3953 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.0633 ± 0.0137 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2971 ± 0.0006 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 2677.7276 ± 197.5750 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 880.7450 ± 28.3412 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.1264 ± 0.0184 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 17.2000 ± 0.7207 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.0992 ± 0.0296 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6793 ± 0.0003 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 3722.4781 ± 110.7387 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 1120.9013 ± 25.7489 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.0841 ± 0.0061 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.5784 ± 0.8237 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.0575 ± 0.0069 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9544 ± 0.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 3686.5771 ± 76.4597 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 1132.4488 ± 10.4695 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.0599 ± 0.0053 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 18.3875 ± 0.1007 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0441 ± 0.0038 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0001 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 5917.8555 ± 201.9331 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 806.8453 ± 10.7070 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.0722 ± 0.0056 | 3 |
| structured_hadamard_float | off (off) | loss | 2.9809 ± 0.0005 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0310 ± 0.0087 | 3 |
| structured_hadamard_float | off (off) | sample_count | 3776.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 16670.0619 ± 268.4730 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 3199.4279 ± 44.2812 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.0563 ± 0.0072 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 18.3074 ± 0.1135 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.0458 ± 0.0043 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 3776.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0001 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 5664.4212 ± 96.1980 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 750.6693 ± 3.2408 | 3 |
| structured_orth_float | off (off) | accuracy | 0.0876 ± 0.0153 | 3 |
| structured_orth_float | off (off) | loss | 2.9855 ± 0.0001 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.0448 ± 0.0160 | 3 |
| structured_orth_float | off (off) | sample_count | 3776.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 17052.5182 ± 417.3055 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 1147.0103 ± 34.6249 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.0576 ± 0.0041 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 18.3228 ± 0.1869 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.0441 ± 0.0046 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 3776.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6821 ± 0.0001 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 5739.9968 ± 127.6391 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 524.1646 ± 2.7788 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.0591 ± 0.0083 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 18.3253 ± 0.2789 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.0478 ± 0.0085 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0001 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 5411.6277 ± 283.8592 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 781.6531 ± 4.0495 | 3 |

## adult (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.5422 ± 0.1061 | 5 |
| backprop_float | off (off) | f1 | 0.4944 ± 0.2483 | 5 |
| backprop_float | off (off) | loss | 0.6915 ± 0.0020 | 5 |
| backprop_float | off (off) | mae | 0.4991 ± 0.0010 | 5 |
| backprop_float | off (off) | precision | 0.7678 ± 0.1111 | 5 |
| backprop_float | off (off) | r2 | -0.0631 ± 0.0042 | 5 |
| backprop_float | off (off) | recall | 0.4425 ± 0.2985 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 50088.4307 ± 6370.6315 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 28740.5647 ± 2570.0715 | 5 |
| dfa_float | off (off) | accuracy | 0.5328 ± 0.1150 | 5 |
| dfa_float | off (off) | f1 | 0.4819 ± 0.2534 | 5 |
| dfa_float | off (off) | loss | 0.6920 ± 0.0031 | 5 |
| dfa_float | off (off) | mae | 0.4994 ± 0.0015 | 5 |
| dfa_float | off (off) | precision | 0.8247 ± 0.1884 | 5 |
| dfa_float | off (off) | r2 | -0.0642 ± 0.0067 | 5 |
| dfa_float | off (off) | recall | 0.4125 ± 0.2557 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 43760.9011 ± 3643.5669 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 30491.0915 ± 1935.6989 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.6438 ± 0.0892 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | f1 | 0.6718 ± 0.1564 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 4.9602 ± 2.4432 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 0.3501 ± 0.0922 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | precision | 0.7681 ± 0.0642 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -0.4158 ± 0.4090 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | recall | 0.6400 ± 0.2198 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6707 ± 0.0072 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 37884.1688 ± 2399.4707 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 27990.4127 ± 3165.2696 | 5 |

## adult (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.8565 ± 0.0014 | 3 |
| backprop_float | off (off) | f1 | 0.6847 ± 0.0028 | 3 |
| backprop_float | off (off) | loss | 0.3237 ± 0.0047 | 3 |
| backprop_float | off (off) | mae | 0.2019 ± 0.0007 | 3 |
| backprop_float | off (off) | precision | 0.7531 ± 0.0039 | 3 |
| backprop_float | off (off) | r2 | 0.4612 ± 0.0034 | 3 |
| backprop_float | off (off) | recall | 0.6277 ± 0.0019 | 3 |
| backprop_float | off (off) | sample_count | 9792.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 41361.0390 ± 856.9266 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 21466.3725 ± 931.3735 | 3 |
| dfa_float | off (off) | accuracy | 0.8595 ± 0.0002 **(best)** | 3 |
| dfa_float | off (off) | f1 | 0.6878 ± 0.0008 | 3 |
| dfa_float | off (off) | loss | 0.3213 ± 0.0023 | 3 |
| dfa_float | off (off) | mae | 0.1939 ± 0.0004 | 3 |
| dfa_float | off (off) | precision | 0.7669 ± 0.0015 | 3 |
| dfa_float | off (off) | r2 | 0.4654 ± 0.0014 | 3 |
| dfa_float | off (off) | recall | 0.6236 ± 0.0019 | 3 |
| dfa_float | off (off) | sample_count | 9792.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 39675.8254 ± 1822.0020 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 23046.8012 ± 499.3270 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.8022 ± 0.0404 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | f1 | 0.3860 ± 0.2913 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 3.9587 ± 0.8783 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 0.1978 ± 0.0406 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | precision | 0.7713 ± 0.0851 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -0.0567 ± 0.2184 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | recall | 0.3192 ± 0.3050 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 9792.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4453 ± 0.0051 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 32489.7553 ± 775.2480 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 19175.7053 ± 208.1112 | 3 |

## ag_news (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.4125 ± 0.0704 | 5 |
| backprop_float | off (off) | loss | 1.3651 ± 0.0082 | 5 |
| backprop_float | off (off) | macro_f1 | 0.3667 ± 0.0635 | 5 |
| backprop_float | off (off) | sample_count | 64.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 4356.2552 ± 945.0397 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 1513.9953 ± 61.4405 | 5 |
| dfa_float | off (off) | accuracy | 0.3812 ± 0.1953 | 5 |
| dfa_float | off (off) | loss | 1.3409 ± 0.0302 | 5 |
| dfa_float | off (off) | macro_f1 | 0.3508 ± 0.1665 | 5 |
| dfa_float | off (off) | sample_count | 64.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 4194.9583 ± 531.1869 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 1447.1737 ± 231.7836 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.4813 ± 0.1840 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 10.1632 ± 3.6991 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.3933 ± 0.2158 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6818 ± 0.0003 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 2755.6920 ± 144.1919 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 863.2277 ± 50.7460 | 5 |

## ag_news (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.9005 ± 0.0007 **(best)** | 2 |
| backprop_float | off (off) | loss | 0.3008 ± 0.0024 | 2 |
| backprop_float | off (off) | macro_f1 | 0.9002 ± 0.0007 | 2 |
| backprop_float | off (off) | sample_count | 25536.0000 | 2 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 2 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 2 |
| backprop_float | off (off) | test_throughput_samples_sec | 3807.2768 ± 243.6141 | 2 |
| backprop_float | off (off) | train_throughput_samples_sec | 1506.6057 ± 36.1746 | 2 |
| dfa_float | off (off) | accuracy | 0.8998 ± 0.0005 | 2 |
| dfa_float | off (off) | loss | 0.3030 ± 0.0016 | 2 |
| dfa_float | off (off) | macro_f1 | 0.8995 ± 0.0005 | 2 |
| dfa_float | off (off) | sample_count | 25536.0000 | 2 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 2 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 2 |
| dfa_float | off (off) | test_throughput_samples_sec | 3378.2391 ± 73.5205 | 2 |
| dfa_float | off (off) | train_throughput_samples_sec | 1411.9319 ± 62.8402 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.7739 ± 0.0013 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 4.4280 ± 0.0572 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.7670 ± 0.0028 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 25536.0000 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6721 ± 0.0008 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 1768.3362 ± 274.1650 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 692.9835 ± 58.0954 | 2 |

## california_housing (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | loss | 6.3994 ± 0.0245 | 3 |
| backprop_float | off (off) | mae | 1.9487 ± 0.0041 | 3 |
| backprop_float | off (off) | r2 | -0.0124 ± 0.0039 | 3 |
| backprop_float | off (off) | rmse | 2.5297 ± 0.0049 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 87757.7091 ± 1022.0328 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 72861.2586 ± 15815.6595 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 122.5007 ± 201.1404 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 4.8773 ± 5.0992 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -18.3791 ± 31.8195 | 3 |
| backprop_ternary_step | ternary (per_step) | rmse | 7.9612 ± 9.4170 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0810 ± 0.0291 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 78312.6326 ± 2059.0049 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 44948.4009 ± 1739.7830 | 3 |
| dfa_float | off (off) | loss | 5.7903 ± 0.3514 | 3 |
| dfa_float | off (off) | mae | 1.8461 ± 0.0570 | 3 |
| dfa_float | off (off) | r2 | 0.0840 ± 0.0556 **(best)** | 3 |
| dfa_float | off (off) | rmse | 2.4055 ± 0.0736 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 82614.6034 ± 950.2247 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 54318.8346 ± 2115.6562 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 9.8164 ± 1.4005 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 2.2261 ± 0.0833 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -0.5529 ± 0.2216 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | rmse | 3.1280 ± 0.2196 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.0626 ± 0.0237 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 83823.6845 ± 515.3488 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 57479.6878 ± 228.9432 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 14681.5489 ± 16828.4125 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 90.7748 ± 83.7806 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -2321.5537 ± 2662.1777 | 3 |
| dfa_ternary_step | ternary (per_step) | rmse | 97.7155 ± 87.7488 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0847 ± 0.0475 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 75960.4744 ± 4689.2680 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 44306.4398 ± 1144.6455 | 3 |
| structured_hadamard_float | off (off) | loss | 6.4155 ± 0.0139 | 3 |
| structured_hadamard_float | off (off) | mae | 1.9511 ± 0.0027 | 3 |
| structured_hadamard_float | off (off) | r2 | -0.0149 ± 0.0022 | 3 |
| structured_hadamard_float | off (off) | rmse | 2.5329 ± 0.0028 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 83001.2409 ± 5342.9101 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 35816.9013 ± 747.0228 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 14.9452 ± 12.2499 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 2.6903 ± 0.9946 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -1.3643 ± 1.9379 | 3 |
| structured_hadamard_ternary | ternary (per_step) | rmse | 3.6629 ± 1.5143 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.2186 ± 0.2010 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 83191.0634 ± 1461.1867 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 26374.2458 ± 560.8930 | 3 |
| structured_orth_float | off (off) | loss | 6.4197 ± 0.0172 | 3 |
| structured_orth_float | off (off) | mae | 1.9518 ± 0.0030 | 3 |
| structured_orth_float | off (off) | r2 | -0.0156 ± 0.0027 | 3 |
| structured_orth_float | off (off) | rmse | 2.5337 ± 0.0034 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 84463.3734 ± 3121.8683 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 26477.3560 ± 345.6982 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 111340.1135 ± 74007.7494 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 309.3594 ± 97.6115 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -17612.4954 ± 11707.6866 | 3 |
| structured_orth_ternary | ternary (per_step) | rmse | 322.3666 ± 105.4982 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.1183 ± 0.0442 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 78221.6017 ± 3178.6024 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 21398.7107 ± 318.4570 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 186.0816 ± 144.5152 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 10.5616 ± 4.5927 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -28.4373 ± 22.8616 | 3 |
| ternary_dfa_step | ternary (per_step) | rmse | 12.9663 ± 5.1900 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.1366 ± 0.0382 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 79211.8284 ± 3961.1406 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 44915.5575 ± 985.4853 | 3 |

## california_housing (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | loss | 0.6816 ± 0.0063 | 5 |
| backprop_float | off (off) | mae | 0.6014 ± 0.0040 | 5 |
| backprop_float | off (off) | r2 | 0.5026 ± 0.0046 | 5 |
| backprop_float | off (off) | rmse | 0.8256 ± 0.0038 | 5 |
| backprop_float | off (off) | sample_count | 4160.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 118330.2065 ± 55901.2774 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 70789.2805 ± 39782.3088 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 22.1981 ± 24.1133 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 2.9896 ± 1.2775 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -15.1987 ± 17.5962 | 3 |
| backprop_ternary_step | ternary (per_step) | rmse | 4.2583 ± 2.4693 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.1291 ± 0.0397 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 137147.7824 ± 6243.4859 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 48291.9444 ± 3572.3750 | 3 |
| dfa_float | off (off) | loss | 0.6083 ± 0.0089 | 5 |
| dfa_float | off (off) | mae | 0.5668 ± 0.0035 | 5 |
| dfa_float | off (off) | r2 | 0.5561 ± 0.0065 | 5 |
| dfa_float | off (off) | rmse | 0.7799 ± 0.0057 | 5 |
| dfa_float | off (off) | sample_count | 4160.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 101040.5107 ± 48428.5303 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 51680.1554 ± 18864.0150 | 5 |
| dfa_float_clip1 | off (off) | loss | 0.7688 ± 0.0135 | 3 |
| dfa_float_clip1 | off (off) | mae | 0.6406 ± 0.0057 | 3 |
| dfa_float_clip1 | off (off) | r2 | 0.4390 ± 0.0099 | 3 |
| dfa_float_clip1 | off (off) | rmse | 0.8768 ± 0.0077 | 3 |
| dfa_float_clip1 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 85934.6571 ± 1406.0270 | 3 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 34962.1641 ± 898.8530 | 3 |
| dfa_float_lr06 | off (off) | loss | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | mae | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | r2 | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | rmse | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 87860.9249 ± 3279.1137 | 3 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 40233.2679 ± 3197.2653 | 3 |
| dfa_float_lr10 | off (off) | loss | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | mae | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | r2 | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | rmse | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 88387.9196 ± 1197.2894 | 3 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 42219.6992 ± 1067.4997 | 3 |
| dfa_float_lr15 | off (off) | loss | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | mae | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | r2 | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | rmse | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 85085.9277 ± 1216.3363 | 3 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 41150.8806 ± 728.9819 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 5.6339 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 2.0648 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -3.1112 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | rmse | 2.3736 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.3566 ± 0.5572 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 125587.4919 ± 8590.1411 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 58359.6210 ± 1088.2825 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 6.0149 ± 0.6599 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | mae | 2.0830 ± 0.0315 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | r2 | -3.3893 ± 0.4816 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | rmse | 2.4501 ± 0.1326 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.3372 ± 0.5740 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 75610.9006 ± 3952.2117 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 35441.2412 ± 169.0688 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 5.6339 ± 0.0001 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 2.0648 ± 0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -3.1112 ± 0.0001 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | rmse | 2.3736 ± 0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 4160.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4231 ± 0.5266 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 63823.8520 ± 22141.1852 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 31151.8436 ± 7429.2039 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 5.6339 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | mae | 2.0648 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | r2 | -3.1112 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | rmse | 2.3736 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.5957 ± 0.3504 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 77665.5150 ± 5829.8415 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 34844.1543 ± 795.2187 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3990.1940 ± 3446.4252 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 47.1905 ± 29.6649 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -2910.7762 ± 2514.9702 | 3 |
| dfa_ternary_step | ternary (per_step) | rmse | 57.4392 ± 32.1932 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.1301 ± 0.0689 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 142183.3301 ± 6749.4762 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 49562.6112 ± 4956.7258 | 3 |
| structured_hadamard_float | off (off) | loss | 0.8640 ± 0.0206 | 3 |
| structured_hadamard_float | off (off) | mae | 0.6732 ± 0.0115 | 3 |
| structured_hadamard_float | off (off) | r2 | 0.3695 ± 0.0151 | 3 |
| structured_hadamard_float | off (off) | rmse | 0.9295 ± 0.0111 | 3 |
| structured_hadamard_float | off (off) | sample_count | 4160.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 161174.3075 ± 1014.8411 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 34423.1477 ± 370.3509 | 3 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.9023 ± 0.0129 | 3 |
| structured_hadamard_float_clip1 | off (off) | mae | 0.6805 ± 0.0101 | 3 |
| structured_hadamard_float_clip1 | off (off) | r2 | 0.3416 ± 0.0094 | 3 |
| structured_hadamard_float_clip1 | off (off) | rmse | 0.9499 ± 0.0068 | 3 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 4160.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 101655.2044 ± 1594.1920 | 3 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 18692.5852 ± 146.8205 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 6131.0257 ± 9574.8019 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 34.6819 ± 37.1107 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -4473.0117 ± 6987.0488 | 3 |
| structured_hadamard_ternary | ternary (per_step) | rmse | 58.0610 ± 64.3422 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 4160.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.2585 ± 0.0987 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 134680.2024 ± 36.7079 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 23979.9197 ± 208.8329 | 3 |
| structured_orth_float | off (off) | loss | 3.0929 ± 0.4880 | 3 |
| structured_orth_float | off (off) | mae | 1.3479 ± 0.1567 | 3 |
| structured_orth_float | off (off) | r2 | -1.2570 ± 0.3561 | 3 |
| structured_orth_float | off (off) | rmse | 1.7551 ± 0.1369 | 3 |
| structured_orth_float | off (off) | sample_count | 4160.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 158996.5665 ± 1363.5662 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 25323.7639 ± 282.6049 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4405.3143 ± 2553.5729 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 43.2086 ± 16.3400 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -3213.7032 ± 1863.4264 | 3 |
| structured_orth_ternary | ternary (per_step) | rmse | 64.6183 ± 18.5657 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 4160.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.1494 ± 0.0035 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 133740.0766 ± 9077.0725 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 20296.9773 ± 481.1351 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 310.2312 ± 417.3877 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 10.7208 ± 7.8813 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -225.3860 ± 304.5816 | 3 |
| ternary_dfa_step | ternary (per_step) | rmse | 14.8690 ± 11.5636 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.1478 ± 0.0520 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 136225.5769 ± 3743.6591 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 47477.8602 ± 3250.4944 | 3 |

## fashion_mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.3975 ± 0.0172 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 3036.0982 ± 825.0856 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 984.2436 ± 515.9198 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 1.0083 ± 0.0425 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 3360.4308 ± 464.3088 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 1325.5441 ± 63.2526 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3975 ± 0.0172 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 3268.8842 ± 404.0974 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 1321.8729 ± 158.9562 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1411 ± 0.0056 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 3381.2607 ± 352.9637 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 1271.9652 ± 138.2842 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.9984 ± 0.0035 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 0.0324 ± 0.0724 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.9986 ± 0.0032 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6140 ± 0.0142 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 2062.8834 ± 456.6302 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 541.7509 ± 81.6254 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0226 ± 0.0006 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 3378.3636 ± 1022.4949 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 1158.0031 ± 296.8677 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.1906 ± 0.1173 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2293 ± 0.0511 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1582 ± 0.0947 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 3746.2850 ± 162.4724 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 1228.4884 ± 119.2169 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0811 ± 0.0021 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 3260.7339 ± 142.0351 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 1421.2542 ± 166.6205 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0226 ± 0.0006 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 2911.2394 ± 367.4455 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 917.0101 ± 353.8001 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0107 ± 0.0003 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 2395.4617 ± 930.0528 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 958.3290 ± 566.6067 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6505 ± 0.0122 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 1723.2262 ± 442.6204 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 858.2472 ± 74.3339 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.9750 ± 0.0559 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 0.5181 ± 1.1585 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.9723 ± 0.0620 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2595 ± 0.0042 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 1563.2680 ± 293.8357 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 722.8483 ± 167.9130 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6505 ± 0.0122 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 1774.2246 ± 357.3737 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 820.4982 ± 107.1525 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.9344 ± 0.0650 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 0.5959 ± 0.6181 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.9081 ± 0.0786 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9533 ± 0.0005 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 1569.3956 ± 277.8921 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 709.3093 ± 93.6045 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8250 ± 0.1095 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5801 ± 2.2889 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7840 ± 0.1208 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6788 ± 0.0011 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 1118.5700 ± 497.0491 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 504.7221 ± 50.4364 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9469 ± 0.0319 | 5 |
| structured_hadamard_float | off (off) | loss | 1.4212 ± 0.0153 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.9190 ± 0.0568 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 3089.2382 ± 401.2475 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 1174.6884 ± 135.2708 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.8703 ± 0.0991 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 1.7797 ± 0.0100 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.8444 ± 0.1005 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 3580.7815 ± 346.4245 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 1083.7025 ± 15.6210 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8469 ± 0.0646 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.1550 ± 1.3458 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7884 ± 0.0782 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6807 ± 0.0006 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 1369.9450 ± 363.9903 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 508.9433 ± 26.7667 | 5 |
| structured_orth_float | off (off) | accuracy | 0.9047 ± 0.0772 | 5 |
| structured_orth_float | off (off) | loss | 1.8305 ± 0.0278 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8788 ± 0.1002 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 3488.8174 ± 399.3791 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 169.3622 ± 47.0925 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8234 ± 0.1449 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 3.6590 ± 3.0019 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7672 ± 0.1338 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6807 ± 0.0006 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 2028.2644 ± 86.9029 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 161.2720 ± 3.1839 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.8891 ± 0.1772 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 2.2990 ± 3.6713 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.8619 ± 0.2093 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6006 ± 0.0032 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 1790.7617 ± 326.0712 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 597.2490 ± 7.9936 | 5 |

## fashion_mnist (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.8703 ± 0.0018 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.3654 ± 0.0032 | 5 |
| backprop_float | off (off) | macro_f1 | 0.8692 ± 0.0018 | 5 |
| backprop_float | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 2236.9175 ± 305.7014 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 1026.3873 ± 78.8558 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.1251 ± 0.0453 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 2.7869 ± 0.5633 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.0472 ± 0.0430 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.4795 ± 0.0188 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 1158.9387 ± 156.4332 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 365.3855 ± 18.9155 | 5 |
| dfa_float | off (off) | accuracy | 0.7786 ± 0.0199 | 5 |
| dfa_float | off (off) | loss | 4.5882 ± 0.4108 | 5 |
| dfa_float | off (off) | macro_f1 | 0.7713 ± 0.0315 | 5 |
| dfa_float | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 4771.8353 ± 1421.4230 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 2351.7166 ± 766.9045 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.5766 ± 0.0852 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 8.7728 ± 1.7669 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.5286 ± 0.0943 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4147 ± 0.0239 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 3317.6086 ± 214.8328 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 1656.2742 ± 98.6179 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.2827 ± 0.0794 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 14.7691 ± 1.6928 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.1982 ± 0.0766 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6394 ± 0.0045 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 3078.7760 ± 215.3636 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 954.1949 ± 76.6851 | 5 |
| structured_orth_float | off (off) | accuracy | 0.1056 ± 0.0031 | 5 |
| structured_orth_float | off (off) | loss | 18.5135 ± 0.0708 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.0201 ± 0.0023 | 5 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 5519.9567 ± 178.3732 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 333.0573 ± 4.8673 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.2047 | 1 |
| structured_orth_ternary | ternary (per_step) | loss | 16.3829 | 1 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.1550 | 1 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 1 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 1 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6709 | 1 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 3565.0845 | 1 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 282.1931 | 1 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.1823 ± 0.0785 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 13.3422 ± 1.2448 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1180 ± 0.0724 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6075 ± 0.0070 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 2854.5976 ± 356.6418 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 869.7190 ± 84.1058 | 5 |

## mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| backprop_float | off (off) | loss | 0.3999 ± 0.0867 | 3 |
| backprop_float | off (off) | macro_f1 | 1.0000 ± 0.0000 | 3 |
| backprop_float | off (off) | sample_count | 128.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 53421.9733 ± 2514.3412 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 30079.9227 ± 1286.9234 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5393 ± 0.0047 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 28655.3822 ± 2542.3364 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 4790.4990 ± 299.1566 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0194 ± 0.0006 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 ± 0.0000 | 3 |
| dfa_float | off (off) | sample_count | 128.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 27950.2131 ± 4290.7979 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 11599.3477 ± 575.0928 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 ± 0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6055 ± 0.0084 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 27701.2024 ± 1168.8808 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 12697.8834 ± 363.6114 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.9089 ± 0.1111 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 1.7141 ± 2.4569 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.9221 ± 0.0739 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6627 ± 0.0106 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 29669.9476 ± 2408.7167 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 5326.9811 ± 69.0649 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.9297 ± 0.0547 | 3 |
| structured_hadamard_float | off (off) | loss | 1.2526 ± 0.0736 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8995 ± 0.0638 | 3 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 51653.8557 ± 3385.9296 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 19231.7020 ± 788.7495 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8021 ± 0.0565 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.1015 ± 1.1712 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7269 ± 0.0067 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6787 ± 0.0005 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 30720.4274 ± 4142.6144 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 4788.1514 ± 71.1208 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7630 ± 0.0798 | 3 |
| structured_orth_float | off (off) | loss | 1.8366 ± 0.0571 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.7358 ± 0.0487 | 3 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 51117.3470 ± 10458.8899 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 5322.2066 ± 123.2764 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8021 ± 0.0565 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 3.8578 ± 0.9988 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7166 ± 0.0130 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6787 ± 0.0004 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 33726.3152 ± 1092.1213 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 2891.8944 ± 32.8496 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 ± 0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5806 ± 0.0082 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 31695.9082 ± 4261.7220 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 5085.7704 ± 99.4644 | 3 |

## mnist (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.9605 ± 0.0011 | 3 |
| backprop_float | off (off) | loss | 0.1348 ± 0.0014 | 3 |
| backprop_float | off (off) | macro_f1 | 0.9603 ± 0.0011 | 3 |
| backprop_float | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 65284.7424 ± 422.4777 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 29882.7215 ± 146.7621 | 3 |
| backprop_float_lr06 | off (off) | accuracy | 0.9511 ± 0.0015 | 3 |
| backprop_float_lr06 | off (off) | loss | 0.1689 ± 0.0014 | 3 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.9507 ± 0.0015 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 44381.4064 ± 1507.2009 | 3 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 20328.0263 ± 1435.3442 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.9605 ± 0.0011 | 3 |
| backprop_float_lr10 | off (off) | loss | 0.1348 ± 0.0014 | 3 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.9603 ± 0.0011 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 47038.5695 ± 726.7801 | 3 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 21246.4038 ± 306.3077 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.9649 ± 0.0001 **(best)** | 3 |
| backprop_float_lr15 | off (off) | loss | 0.1164 ± 0.0015 | 3 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.9647 ± 0.0001 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 46463.8410 ± 1328.7509 | 3 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 20879.9020 ± 628.8109 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.4042 ± 0.2111 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 8.9116 ± 2.1900 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3741 ± 0.2313 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5601 ± 0.0089 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 30871.8488 ± 890.9804 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 4331.3029 ± 18.8356 | 3 |
| dfa_float | off (off) | accuracy | 0.9254 ± 0.0081 | 3 |
| dfa_float | off (off) | loss | 1.4171 ± 0.1924 | 3 |
| dfa_float | off (off) | macro_f1 | 0.9250 ± 0.0085 | 3 |
| dfa_float | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 31213.0295 ± 476.0122 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 11299.4667 ± 65.6092 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.9514 ± 0.0010 | 3 |
| dfa_float_lr06 | off (off) | loss | 0.1633 ± 0.0039 | 3 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.9510 ± 0.0011 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 22355.5933 ± 304.4425 | 3 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 8404.9948 ± 111.4277 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.9254 ± 0.0081 | 3 |
| dfa_float_lr10 | off (off) | loss | 1.4171 ± 0.1924 | 3 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.9250 ± 0.0085 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 22215.5780 ± 289.7364 | 3 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 8304.6800 ± 124.8639 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.9358 ± 0.0017 | 3 |
| dfa_float_lr15 | off (off) | loss | 1.3231 ± 0.0385 | 3 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.9354 ± 0.0019 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 21164.1829 ± 818.6561 | 3 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 7362.4313 ± 391.7822 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.7607 ± 0.0731 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 4.9544 ± 1.5160 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.7456 ± 0.0966 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4789 ± 0.0024 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 30348.9562 ± 122.6117 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 10988.2492 ± 73.8989 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.7671 ± 0.0598 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 4.8219 ± 1.2429 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.7539 ± 0.0681 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.1847 ± 0.0005 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 14310.4602 ± 4408.2872 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 5276.4333 ± 1361.8096 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.7607 ± 0.0731 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 4.9544 ± 1.5160 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.7456 ± 0.0966 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4789 ± 0.0024 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 15902.6137 ± 3818.4528 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 5163.0055 ± 1362.5661 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.7935 ± 0.0085 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.2718 ± 0.1766 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.7928 ± 0.0095 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.7472 ± 0.0051 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 18260.1413 ± 1026.1935 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 6185.1190 ± 123.3194 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.1756 ± 0.0839 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 17.0506 ± 1.7745 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0879 ± 0.0655 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6334 ± 0.0019 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 30359.7975 ± 910.3169 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 4527.5008 ± 50.7236 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.9299 ± 0.0009 | 3 |
| structured_hadamard_float | off (off) | loss | 0.2347 ± 0.0016 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.9295 ± 0.0009 | 3 |
| structured_hadamard_float | off (off) | sample_count | 14016.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 65656.5516 ± 1282.5006 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 18194.3900 ± 70.7654 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.3533 ± 0.0617 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 13.2049 ± 1.3390 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2767 ± 0.0640 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 14016.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6774 ± 0.0004 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 31016.9338 ± 388.6253 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 4203.2317 ± 40.2755 | 3 |
| structured_orth_float | off (off) | accuracy | 0.8257 ± 0.0035 | 3 |
| structured_orth_float | off (off) | loss | 0.6278 ± 0.0123 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.8227 ± 0.0035 | 3 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 64151.0800 ± 1188.4003 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 4526.3946 ± 190.8847 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.3391 ± 0.0318 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 13.5022 ± 0.6828 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.2589 ± 0.0202 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6773 ± 0.0004 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 30994.1801 ± 364.6805 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 2455.7255 ± 12.5928 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.3005 ± 0.0504 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 14.4632 ± 1.0474 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1828 ± 0.0449 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5895 ± 0.0039 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 30590.2373 ± 746.1712 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 4426.5397 ± 22.9964 | 3 |

## ucr (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| backprop_float | off (off) | loss | 1.1311 ± 0.0485 | 3 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 32.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 50335.9869 ± 3706.2032 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 30230.1368 ± 1396.8363 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6118 ± 0.0295 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 47157.3277 ± 5397.0612 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 16273.0870 ± 1274.6365 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0725 ± 0.0120 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 32.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 47549.6669 ± 8772.7309 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 26300.9210 ± 4319.1058 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6373 ± 0.0091 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 49291.8923 ± 1852.5927 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 28919.1458 ± 4002.2842 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8177 ± 0.1263 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5722 ± 2.7832 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7355 ± 0.1863 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6647 ± 0.0080 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 51097.3766 ± 3135.3668 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 18415.9402 ± 2849.0815 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.6771 ± 0.4045 | 3 |
| structured_hadamard_float | off (off) | loss | 1.3163 ± 0.0194 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.6049 ± 0.4360 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 48358.2107 ± 6883.1411 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 17406.3199 ± 356.2727 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.2094 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.6941 ± 0.0074 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6772 ± 0.0021 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 48343.7547 ± 8086.4024 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 11160.9440 ± 677.6739 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7760 ± 0.2189 | 3 |
| structured_orth_float | off (off) | loss | 1.3158 ± 0.0444 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.6657 ± 0.3210 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 44240.0328 ± 7437.7879 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 8314.5240 ± 127.0926 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4.1640 ± 0.0786 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.6772 ± 0.0199 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6770 ± 0.0022 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 42631.4021 ± 4540.2041 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 6929.4500 ± 488.3038 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6072 ± 0.0064 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 51612.2241 ± 3049.0575 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 17675.4828 ± 1360.0697 | 3 |

## ucr (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| backprop_float | off (off) | loss | 1.0295 ± 0.0596 | 3 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 32.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 41105.7350 ± 1381.6348 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 28270.9856 ± 1799.6195 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6118 ± 0.0295 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 38061.0459 ± 1521.4688 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 13778.8572 ± 1334.5184 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0327 ± 0.0041 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 32.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 37193.7595 ± 1903.4467 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 22665.6426 ± 884.4934 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6373 ± 0.0091 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 37568.9243 ± 704.0968 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 22776.9334 ± 922.5723 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8177 ± 0.1263 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5722 ± 2.7832 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7355 ± 0.1863 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6647 ± 0.0080 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 36351.2168 ± 3146.8542 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 14879.8923 ± 992.9987 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.7969 ± 0.2204 | 3 |
| structured_hadamard_float | off (off) | loss | 1.2706 ± 0.0087 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.6838 ± 0.3273 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 38488.5719 ± 4039.5525 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 13662.3827 ± 485.2378 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.2094 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.6941 ± 0.0074 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6772 ± 0.0021 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 39054.8162 ± 2524.1131 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 10237.9754 ± 784.3918 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7760 ± 0.2189 | 3 |
| structured_orth_float | off (off) | loss | 1.2960 ± 0.0434 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.6644 ± 0.3211 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 41204.3096 ± 1394.5144 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 6680.8939 ± 31.7956 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4.1640 ± 0.0786 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.6772 ± 0.0199 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6770 ± 0.0022 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 34852.1274 ± 4936.3889 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 5405.9354 ± 402.0860 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6072 ± 0.0064 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 34596.8636 ± 7185.5088 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 15101.9448 ± 241.5737 | 3 |

## Recommendations

- Vision (MNIST, real): backprop_float with lr≈0.075 leads accuracy; if ternary forward is required, use DFA with per_epoch flips and τ≈0.05.

- Text (20 Newsgroups, real): DFA float (lr≈0.05) remains most stable; ternary benefits from per_epoch flips with τ in [0.02, 0.05], accepting reduced accuracy for higher sparsity.

- Tabular (California Housing, real): DFA float with lr≈0.05 and grad_clip=1.0

  improves robustness;

  structured hadamard float is the throughput leader. Avoid per_step ternary.

- Time-series (UCR): accuracy saturates at 1.0 across methods; prefer ternary DFA

  for deployability and footprint.

- Flip scheduling: prefer per_epoch over per_step on non-vision modalities.

- Ternary threshold: start at τ=0.05 and adjust by modality (lower for text).
