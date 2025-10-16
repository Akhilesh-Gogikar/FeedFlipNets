# FeedFlipNets Benchmark Summary

Aggregated over seeds with mean ± 95% CI.


## Topline Highlights

| Dataset | Mode | Metric | Mean ± 95% CI | Strategy Variant | Flip | n |
|---|---|---|---|---|---|---:|
| 20newsgroups | offline | Accuracy | 0.3906 ± 0.1692 | backprop_ternary_step | ternary (per_step) | 3 |
| 20newsgroups | offline | Macro-F1 | 0.3414 ± 0.1177 | backprop_ternary_step | ternary (per_step) | 3 |
| 20newsgroups | offline | Zero Ratio | 0.6819 ± 0.0002 | structured_orth_ternary | ternary (per_step) | 3 |
| 20newsgroups | offline | Test Throughput (samples/s) | 17760.0556 ± 1388.3168 | backprop_float | off (off) | 3 |
| 20newsgroups | real | Accuracy | 0.3206 ± 0.0113 | dfa_float_lr15 | off (off) | 3 |
| 20newsgroups | real | Macro-F1 | 0.2744 ± 0.0111 | dfa_float_lr15 | off (off) | 3 |
| 20newsgroups | real | Zero Ratio | 0.9544 ± 0.0001 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| 20newsgroups | real | Test Throughput (samples/s) | 17155.5581 ± 557.4105 | backprop_float | off (off) | 3 |
| adult | offline | Accuracy | 0.9089 ± 0.0784 | dfa_float_lr15 | off (off) | 3 |
| adult | offline | R² | 0.6091 ± 0.0705 | dfa_float_lr15 | off (off) | 3 |
| adult | offline | Zero Ratio | 0.9553 ± 0.0037 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| adult | offline | Test Throughput (samples/s) | 50088.4307 ± 7910.1837 | backprop_float | off (off) | 5 |
| adult | real | Accuracy | 0.8595 ± 0.0006 | dfa_float | off (off) | 3 |
| adult | real | R² | 0.4654 ± 0.0036 | dfa_float | off (off) | 3 |
| adult | real | Zero Ratio | 0.4453 ± 0.0126 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 3 |
| adult | real | Test Throughput (samples/s) | 41361.0390 ± 2128.7236 | backprop_float | off (off) | 3 |
| ag_news | offline | Accuracy | 0.4813 ± 0.2285 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Macro-F1 | 0.3933 ± 0.2679 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Zero Ratio | 0.6818 ± 0.0004 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Test Throughput (samples/s) | 4356.2552 ± 1173.4218 | backprop_float | off (off) | 5 |
| ag_news | real | Accuracy | 0.9005 ± 0.0060 | backprop_float | off (off) | 2 |
| ag_news | real | Macro-F1 | 0.9002 ± 0.0062 | backprop_float | off (off) | 2 |
| ag_news | real | Zero Ratio | 0.6721 ± 0.0075 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 2 |
| ag_news | real | Test Throughput (samples/s) | 3807.2768 ± 2188.7856 | backprop_float | off (off) | 2 |
| california_housing | offline | R² | 0.0840 ± 0.1381 | dfa_float | off (off) | 3 |
| california_housing | offline | Zero Ratio | 0.2186 ± 0.4993 | structured_hadamard_ternary | ternary (per_step) | 3 |
| california_housing | offline | Test Throughput (samples/s) | 87757.7091 ± 2538.8701 | backprop_float | off (off) | 3 |
| california_housing | real | R² | 0.5561 ± 0.0081 | dfa_float | off (off) | 5 |
| california_housing | real | Zero Ratio | 0.5957 ± 0.8704 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| california_housing | real | Test Throughput (samples/s) | 161174.3075 ± 2521.0051 | structured_hadamard_float | off (off) | 3 |
| cifar10 | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| cifar10 | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| cifar10 | offline | Zero Ratio | 0.9535 ± 0.0007 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| cifar10 | offline | Test Throughput (samples/s) | 230.6713 ± 19.8766 | dfa_float_clip1 | off (off) | 5 |
| fashion_mnist | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| fashion_mnist | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| fashion_mnist | offline | Zero Ratio | 0.9533 ± 0.0006 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| fashion_mnist | offline | Test Throughput (samples/s) | 189.7377 ± 34.5414 | dfa_float | off (off) | 5 |
| fashion_mnist | real | Accuracy | 0.8738 ± 0.0032 | backprop_float_lr15 | off (off) | 5 |
| fashion_mnist | real | Macro-F1 | 0.8729 ± 0.0031 | backprop_float_lr15 | off (off) | 5 |
| fashion_mnist | real | Zero Ratio | 0.7332 ± 0.0092 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| fashion_mnist | real | Test Throughput (samples/s) | 366.3145 ± 35.5270 | dfa_float_clip1 | off (off) | 5 |
| mnist | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| mnist | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| mnist | offline | Zero Ratio | 0.9501 ± 0.0016 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| mnist | offline | Test Throughput (samples/s) | 329.5882 ± 57.1488 | structured_hadamard_float_clip1 | off (off) | 5 |
| mnist | real | Accuracy | 0.9646 ± 0.0004 | backprop_float_lr15 | off (off) | 5 |
| mnist | real | Macro-F1 | 0.9645 ± 0.0004 | backprop_float_lr15 | off (off) | 5 |
| mnist | real | Zero Ratio | 0.7480 ± 0.0044 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| mnist | real | Test Throughput (samples/s) | 245.2276 ± 16.4256 | backprop_float | off (off) | 5 |
| ucr | offline | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | offline | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | offline | Zero Ratio | 0.6772 ± 0.0052 | structured_hadamard_ternary | ternary (per_step) | 3 |
| ucr | offline | Test Throughput (samples/s) | 51612.2241 ± 7574.2786 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Zero Ratio | 0.6772 ± 0.0052 | structured_hadamard_ternary | ternary (per_step) | 3 |
| ucr | real | Test Throughput (samples/s) | 41204.3096 ± 3464.1658 | structured_orth_float | off (off) | 3 |

## Best Configs

| Dataset | Mode | Primary | Best (μ±95% CI) | Variant | Flip | n | Baseline (μ±σ) | Δ | Effect Size |
|---|---|---|---|---|---|---:|---|---:|---:|
| 20newsgroups | offline | accuracy | 0.2083 ± 0.2138 | structured_orth_ternary | ternary (per_step) | 3 | 0.1771 ± 0.0722 | 0.0312 | 0.393 |
| 20newsgroups | real | accuracy | 0.0576 ± 0.0102 | structured_orth_ternary | ternary (per_step) | 3 | 0.3206 ± 0.0046 | -0.2630 | -60.619 |
| adult | offline | accuracy | 0.5130 ± 0.2907 | structured_orth_ternary | ternary (per_step) | 3 | 0.9089 ± 0.0316 | -0.3958 | -4.619 |
| adult | real | accuracy | 0.8565 ± 0.0035 | backprop_float | off (off) | 3 | 0.8595 ± 0.0002 | -0.0030 | -3.004 |
| ag_news | offline | accuracy | 0.4125 ± 0.0874 | backprop_float | off (off) | 5 | 0.4125 ± 0.0704 | 0.0000 | 0.000 |
| ag_news | real | accuracy | 0.9005 ± 0.0060 | backprop_float | off (off) | 2 | 0.9005 ± 0.0007 | 0.0000 | 0.000 |
| california_housing | offline | r2 | -17612.4954 ± 29083.5058 | structured_orth_ternary | ternary (per_step) | 3 | 0.0840 ± 0.0556 | -17612.5794 | -2.127 |
| california_housing | real | r2 | -3213.7032 ± 4629.0077 | structured_orth_ternary | ternary (per_step) | 3 | 0.5561 ± 0.0065 | -3214.2592 | -2.439 |
| cifar10 | offline | accuracy | 0.8250 ± 0.1094 | structured_orth_ternary | ternary (per_step) | 5 | 1.0000 | -0.1750 | -2.808 |
| fashion_mnist | offline | accuracy | 0.8469 ± 0.1028 | structured_orth_ternary | ternary (per_step) | 5 | 1.0000 | -0.1531 | -2.616 |
| fashion_mnist | real | accuracy | 0.3705 ± 0.0523 | structured_orth_ternary | ternary (per_step) | 5 | 0.8738 ± 0.0026 | -0.5034 | -16.880 |
| mnist | offline | accuracy | 0.8547 ± 0.0600 | structured_orth_ternary | ternary (per_step) | 5 | 1.0000 | -0.1453 | -4.250 |
| mnist | real | accuracy | 0.3603 ± 0.0447 | structured_orth_ternary | ternary (per_step) | 5 | 0.9646 ± 0.0003 | -0.6043 | -23.750 |
| ucr | offline | accuracy | 0.7969 | structured_orth_ternary | ternary (per_step) | 3 | 1.0000 | -0.2031 | 0.000 |
| ucr | real | accuracy | 0.7969 | structured_orth_ternary | ternary (per_step) | 3 | 1.0000 | -0.2031 | 0.000 |

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
| backprop_float_lr06 | off (off) | accuracy | 0.5859 ± 0.1681 | 3 |
| backprop_float_lr06 | off (off) | f1 | 0.5349 ± 0.3495 | 3 |
| backprop_float_lr06 | off (off) | loss | 0.6909 ± 0.0027 | 3 |
| backprop_float_lr06 | off (off) | mae | 0.4988 ± 0.0014 | 3 |
| backprop_float_lr06 | off (off) | precision | 0.8269 ± 0.1689 | 3 |
| backprop_float_lr06 | off (off) | r2 | -0.0618 ± 0.0059 | 3 |
| backprop_float_lr06 | off (off) | recall | 0.5083 ± 0.3804 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 16509.8730 ± 973.7512 | 3 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 7049.7947 ± 278.3565 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.5833 ± 0.1641 | 3 |
| backprop_float_lr10 | off (off) | f1 | 0.5227 ± 0.3373 | 3 |
| backprop_float_lr10 | off (off) | loss | 0.6900 ± 0.0029 | 3 |
| backprop_float_lr10 | off (off) | mae | 0.4984 ± 0.0014 | 3 |
| backprop_float_lr10 | off (off) | precision | 0.8527 ± 0.1671 | 3 |
| backprop_float_lr10 | off (off) | r2 | -0.0600 ± 0.0062 | 3 |
| backprop_float_lr10 | off (off) | recall | 0.4667 ± 0.3401 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 13733.8732 ± 2469.2006 | 3 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 6000.0081 ± 1650.4265 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.5938 ± 0.1663 | 3 |
| backprop_float_lr15 | off (off) | f1 | 0.5079 ± 0.3426 | 3 |
| backprop_float_lr15 | off (off) | loss | 0.6889 ± 0.0032 | 3 |
| backprop_float_lr15 | off (off) | mae | 0.4978 ± 0.0016 | 3 |
| backprop_float_lr15 | off (off) | precision | 0.8993 ± 0.1077 | 3 |
| backprop_float_lr15 | off (off) | r2 | -0.0576 ± 0.0068 | 3 |
| backprop_float_lr15 | off (off) | recall | 0.4208 ± 0.3143 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 14551.9545 ± 4634.5867 | 3 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 5780.4851 ± 1148.9821 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.5339 ± 0.1308 | 3 |
| backprop_ternary_step | ternary (per_step) | f1 | 0.5458 ± 0.1986 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 7.2513 ± 1.5158 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 0.4655 ± 0.1178 | 3 |
| backprop_ternary_step | ternary (per_step) | precision | 0.6675 ± 0.1563 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -0.9188 ± 0.4841 | 3 |
| backprop_ternary_step | ternary (per_step) | recall | 0.4917 ± 0.2629 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6801 ± 0.0066 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 12270.1506 ± 2279.0736 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 4282.6546 ± 1064.4825 | 3 |
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
| dfa_float_clip1 | off (off) | accuracy | 0.5208 ± 0.1066 | 3 |
| dfa_float_clip1 | off (off) | f1 | 0.4966 ± 0.3297 | 3 |
| dfa_float_clip1 | off (off) | loss | 0.6926 ± 0.0016 | 3 |
| dfa_float_clip1 | off (off) | mae | 0.4997 ± 0.0008 | 3 |
| dfa_float_clip1 | off (off) | precision | 0.7072 ± 0.1157 | 3 |
| dfa_float_clip1 | off (off) | r2 | -0.0654 ± 0.0034 | 3 |
| dfa_float_clip1 | off (off) | recall | 0.5125 ± 0.3899 | 3 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 15734.6856 ± 1942.1155 | 3 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 6356.7891 ± 138.0723 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.5391 ± 0.1918 | 3 |
| dfa_float_lr06 | off (off) | f1 | 0.3766 ± 0.4104 | 3 |
| dfa_float_lr06 | off (off) | loss | 0.6853 ± 0.0053 | 3 |
| dfa_float_lr06 | off (off) | mae | 0.4958 ± 0.0027 | 3 |
| dfa_float_lr06 | off (off) | precision | 0.5870 ± 0.5222 | 3 |
| dfa_float_lr06 | off (off) | r2 | -0.0499 ± 0.0112 | 3 |
| dfa_float_lr06 | off (off) | recall | 0.3542 ± 0.4607 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 13714.5840 ± 2089.2449 | 3 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 7155.2834 ± 195.5314 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.8229 ± 0.0369 | 3 |
| dfa_float_lr10 | off (off) | f1 | 0.8364 ± 0.0401 | 3 |
| dfa_float_lr10 | off (off) | loss | 0.5466 ± 0.0190 | 3 |
| dfa_float_lr10 | off (off) | mae | 0.4125 ± 0.0122 | 3 |
| dfa_float_lr10 | off (off) | precision | 0.9831 ± 0.0014 | 3 |
| dfa_float_lr10 | off (off) | r2 | 0.2276 ± 0.0370 | 3 |
| dfa_float_lr10 | off (off) | recall | 0.7292 ± 0.0591 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 15613.2733 ± 832.0295 | 3 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 7273.5395 ± 198.7706 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.9089 ± 0.0316 **(best)** | 3 |
| dfa_float_lr15 | off (off) | f1 | 0.9219 ± 0.0292 | 3 |
| dfa_float_lr15 | off (off) | loss | 0.3217 ± 0.0189 | 3 |
| dfa_float_lr15 | off (off) | mae | 0.2527 ± 0.0148 | 3 |
| dfa_float_lr15 | off (off) | precision | 0.9858 ± 0.0008 | 3 |
| dfa_float_lr15 | off (off) | r2 | 0.6091 ± 0.0284 | 3 |
| dfa_float_lr15 | off (off) | recall | 0.8667 ± 0.0505 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 16535.0194 ± 939.4538 | 3 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 6897.2037 ± 385.6870 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.6094 ± 0.0921 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | f1 | 0.6171 ± 0.1880 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 6.3082 ± 2.2061 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 0.3863 ± 0.0976 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | precision | 0.7748 ± 0.0830 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -0.5807 ± 0.4423 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | recall | 0.5667 ± 0.2694 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6687 ± 0.0094 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 12622.7968 ± 1864.7068 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 6000.3469 ± 162.7120 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.5130 ± 0.0942 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | f1 | 0.5006 ± 0.2813 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 9.0169 ± 1.8277 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | mae | 0.4827 ± 0.0938 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | precision | 0.6740 ± 0.0690 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | r2 | -1.0430 ± 0.4043 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | recall | 0.4917 ± 0.3429 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2648 ± 0.0151 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 15295.4422 ± 628.2294 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 6093.4874 ± 876.5819 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.6438 ± 0.0892 | 5 |
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
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.5599 ± 0.0597 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | f1 | 0.6299 ± 0.1758 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 0.8439 ± 0.2429 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | mae | 0.4744 ± 0.0632 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | precision | 0.6677 ± 0.0929 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | r2 | -0.2191 ± 0.3520 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | recall | 0.7000 ± 0.3606 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9553 ± 0.0015 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 14601.5060 ± 1297.4506 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 5919.9562 ± 186.8185 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.5339 ± 0.0655 | 3 |
| dfa_ternary_step | ternary (per_step) | f1 | 0.5984 ± 0.1102 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 7.0006 ± 0.8430 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 0.4691 ± 0.0728 | 3 |
| dfa_ternary_step | ternary (per_step) | precision | 0.6376 ± 0.0148 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -0.9352 ± 0.2900 | 3 |
| dfa_ternary_step | ternary (per_step) | recall | 0.5833 ± 0.2032 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6798 ± 0.0068 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 16215.4888 ± 246.5072 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 5738.1191 ± 117.9449 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.5625 ± 0.1484 | 3 |
| structured_hadamard_float | off (off) | f1 | 0.5263 ± 0.3413 | 3 |
| structured_hadamard_float | off (off) | loss | 0.6919 ± 0.0026 | 3 |
| structured_hadamard_float | off (off) | mae | 0.4993 ± 0.0013 | 3 |
| structured_hadamard_float | off (off) | precision | 0.7479 ± 0.1088 | 3 |
| structured_hadamard_float | off (off) | r2 | -0.0640 ± 0.0055 | 3 |
| structured_hadamard_float | off (off) | recall | 0.5250 ± 0.3947 | 3 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 16710.3283 ± 616.3128 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 4548.9548 ± 378.8141 | 3 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.5625 ± 0.1484 | 3 |
| structured_hadamard_float_clip1 | off (off) | f1 | 0.5263 ± 0.3413 | 3 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.6919 ± 0.0026 | 3 |
| structured_hadamard_float_clip1 | off (off) | mae | 0.4993 ± 0.0013 | 3 |
| structured_hadamard_float_clip1 | off (off) | precision | 0.7479 ± 0.1088 | 3 |
| structured_hadamard_float_clip1 | off (off) | r2 | -0.0640 ± 0.0055 | 3 |
| structured_hadamard_float_clip1 | off (off) | recall | 0.5250 ± 0.3947 | 3 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 16474.3741 ± 1216.3022 | 3 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 4388.2189 ± 382.4747 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.5365 ± 0.1130 | 3 |
| structured_hadamard_ternary | ternary (per_step) | f1 | 0.5623 ± 0.1570 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 6.9442 ± 0.7877 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 0.4694 ± 0.1087 | 3 |
| structured_hadamard_ternary | ternary (per_step) | precision | 0.6749 ± 0.1377 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -0.9301 ± 0.4382 | 3 |
| structured_hadamard_ternary | ternary (per_step) | recall | 0.5083 ± 0.2306 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6802 ± 0.0064 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 12857.2240 ± 1981.3930 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 3639.7207 ± 319.5973 | 3 |
| structured_orth_float | off (off) | accuracy | 0.5599 ± 0.1524 | 3 |
| structured_orth_float | off (off) | f1 | 0.5191 ± 0.3536 | 3 |
| structured_orth_float | off (off) | loss | 0.6919 ± 0.0026 | 3 |
| structured_orth_float | off (off) | mae | 0.4994 ± 0.0013 | 3 |
| structured_orth_float | off (off) | precision | 0.7400 ± 0.0971 | 3 |
| structured_orth_float | off (off) | r2 | -0.0641 ± 0.0056 | 3 |
| structured_orth_float | off (off) | recall | 0.5208 ± 0.4018 | 3 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 16897.0019 ± 666.4320 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 5426.9110 ± 646.5914 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.5130 ± 0.1170 | 3 |
| structured_orth_ternary | ternary (per_step) | f1 | 0.5293 ± 0.1934 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 7.3553 ± 1.4238 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 0.4775 ± 0.1113 | 3 |
| structured_orth_ternary | ternary (per_step) | precision | 0.6322 ± 0.1066 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -0.9578 ± 0.4574 | 3 |
| structured_orth_ternary | ternary (per_step) | recall | 0.4833 ± 0.2641 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6802 ± 0.0065 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 9940.3261 ± 4615.9916 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 3716.7583 ± 1126.2360 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.5521 ± 0.1201 | 3 |
| ternary_dfa_step | ternary (per_step) | f1 | 0.5936 ± 0.1946 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 7.2773 ± 1.0543 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 0.4549 ± 0.1240 | 3 |
| ternary_dfa_step | ternary (per_step) | precision | 0.6488 ± 0.0842 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -0.8996 ± 0.4879 | 3 |
| ternary_dfa_step | ternary (per_step) | recall | 0.5792 ± 0.2757 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6800 ± 0.0062 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 14710.6294 ± 1533.1334 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 5437.1334 ± 380.1660 | 3 |

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
| dfa_float | off (off) | r2 | 0.5561 ± 0.0065 **(best)** | 5 |
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
| dfa_float_lr06 | off (off) | loss | 5.6339 | 1 |
| dfa_float_lr06 | off (off) | mae | 2.0648 | 1 |
| dfa_float_lr06 | off (off) | r2 | -3.1112 | 1 |
| dfa_float_lr06 | off (off) | rmse | 2.3736 | 1 |
| dfa_float_lr06 | off (off) | sample_count | 4160.0000 | 1 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 1 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 1 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 89112.8630 | 1 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 42729.9537 | 1 |
| dfa_float_lr10 | off (off) | loss | 5.6331 ± 0.0012 | 2 |
| dfa_float_lr10 | off (off) | mae | 2.0647 ± 0.0002 | 2 |
| dfa_float_lr10 | off (off) | r2 | -3.1106 ± 0.0008 | 2 |
| dfa_float_lr10 | off (off) | rmse | 2.3734 ± 0.0002 | 2 |
| dfa_float_lr10 | off (off) | sample_count | 4160.0000 | 2 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 2 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 2 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 87826.7945 ± 988.8558 | 2 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 42703.2768 ± 935.9633 | 2 |
| dfa_float_lr15 | off (off) | loss | 5.6339 | 1 |
| dfa_float_lr15 | off (off) | mae | 2.0648 | 1 |
| dfa_float_lr15 | off (off) | r2 | -3.1112 | 1 |
| dfa_float_lr15 | off (off) | rmse | 2.3736 | 1 |
| dfa_float_lr15 | off (off) | sample_count | 4160.0000 | 1 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 1 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 1 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 85521.2626 | 1 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 40522.3364 | 1 |
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

## cifar10 (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.2362 ± 0.0251 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 214.1708 ± 26.4604 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 98.2057 ± 2.5268 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 0.6403 ± 0.0671 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 215.3774 ± 19.6375 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 98.2120 ± 2.3263 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.2362 ± 0.0251 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 223.0111 ± 40.2430 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 100.6332 ± 5.0838 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1016 ± 0.0078 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 211.9118 ± 19.6823 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 101.2210 ± 4.0554 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 5 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6348 ± 0.0084 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 82.8979 ± 4.4618 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 51.9163 ± 1.8371 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0231 ± 0.0006 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 198.2328 ± 19.2064 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 105.0146 ± 3.2666 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.1250 ± 0.0688 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2824 ± 0.0861 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1013 ± 0.0506 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 230.6713 ± 16.0080 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 99.7528 ± 4.2208 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0809 ± 0.0050 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 223.9662 ± 25.6657 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 104.2739 ± 6.5166 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0231 ± 0.0006 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 220.9940 ± 15.6790 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 104.1663 ± 1.9093 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0109 ± 0.0003 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 201.4673 ± 20.2193 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 103.8049 ± 6.0854 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.9781 ± 0.0407 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 0.4533 ± 0.8444 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.9650 ± 0.0702 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6461 ± 0.0053 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 84.1402 ± 5.8435 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 61.7503 ± 2.2662 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2774 ± 0.0018 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 84.3464 ± 8.4611 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 64.0042 ± 2.8319 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.9781 ± 0.0407 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 0.4533 ± 0.8444 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.9650 ± 0.0702 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6461 ± 0.0053 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 87.3592 ± 4.6065 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 63.0210 ± 1.7959 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.8688 ± 0.0959 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 2.3918 ± 1.8822 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.8590 ± 0.1156 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9535 ± 0.0006 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 82.7078 ± 2.9104 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 62.5935 ± 3.3641 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.9266 ± 0.1344 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 1.5219 ± 2.7845 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.8722 ± 0.2159 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6765 ± 0.0036 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 83.9920 ± 11.7805 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 55.2254 ± 2.1611 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9187 ± 0.0726 | 5 |
| structured_hadamard_float | off (off) | loss | 1.5590 ± 0.0816 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8819 ± 0.1084 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 195.5261 ± 28.3228 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 101.3487 ± 5.6666 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.7875 ± 0.0813 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 1.8579 ± 0.0725 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.7259 ± 0.1043 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 204.5050 ± 19.9146 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 100.0662 ± 4.8694 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8516 ± 0.0767 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.0761 ± 1.5904 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7763 ± 0.0733 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6814 ± 0.0004 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 82.3866 ± 5.9112 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 54.7997 ± 1.5354 | 5 |
| structured_orth_float | off (off) | accuracy | 0.9187 ± 0.0767 | 5 |
| structured_orth_float | off (off) | loss | 1.5603 ± 0.0773 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8756 ± 0.1181 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 224.2679 ± 27.3321 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 102.1367 ± 8.0348 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8250 ± 0.0881 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 3.6266 ± 1.8267 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7609 ± 0.0715 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6814 ± 0.0004 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 77.6485 ± 14.2870 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 55.9499 ± 1.8016 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.8750 ± 0.1075 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 2.5904 ± 2.2287 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.8256 ± 0.1128 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6344 ± 0.0031 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 86.5054 ± 9.2460 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 56.1488 ± 0.8659 | 5 |

## fashion_mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.3975 ± 0.0172 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 162.4573 ± 15.6706 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 90.6341 ± 6.0860 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 1.0083 ± 0.0425 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 176.3951 ± 13.3938 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 76.8736 ± 2.8981 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3975 ± 0.0172 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 163.0843 ± 16.0591 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 75.3293 ± 5.7411 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1411 ± 0.0056 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 169.4661 ± 14.4957 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 71.0757 ± 7.7600 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.9984 ± 0.0035 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 0.0324 ± 0.0724 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.9986 ± 0.0032 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6140 ± 0.0142 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 95.5624 ± 3.2760 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 60.8838 ± 1.2550 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0226 ± 0.0006 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 189.7377 ± 27.8186 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 92.2744 ± 14.7343 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.1906 ± 0.1173 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2293 ± 0.0511 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1582 ± 0.0947 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 162.3893 ± 19.6423 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 76.4787 ± 3.4267 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0811 ± 0.0021 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 170.0911 ± 10.9507 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 73.3401 ± 7.2651 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0226 ± 0.0006 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 173.2018 ± 12.6049 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 70.5178 ± 6.9988 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0107 ± 0.0003 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 181.8000 ± 15.2645 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 74.8880 ± 5.3024 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6505 ± 0.0122 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 96.6668 ± 6.3845 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 68.3758 ± 3.3994 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.9750 ± 0.0559 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 0.5181 ± 1.1585 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.9723 ± 0.0620 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2595 ± 0.0042 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 74.6788 ± 7.1702 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 47.6521 ± 4.2995 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6505 ± 0.0122 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 75.2829 ± 7.4708 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 52.3841 ± 0.6596 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.9344 ± 0.0650 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 0.5959 ± 0.6181 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.9081 ± 0.0786 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9533 ± 0.0005 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 78.6530 ± 2.7424 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 53.2933 ± 1.5785 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8250 ± 0.1095 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5801 ± 2.2889 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7840 ± 0.1208 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6788 ± 0.0011 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 89.7688 ± 4.6512 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 62.8128 ± 3.1368 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9031 ± 0.0730 | 5 |
| structured_hadamard_float | off (off) | loss | 1.8252 ± 0.0344 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8789 ± 0.0959 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 149.8383 ± 13.3418 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 74.8377 ± 4.6004 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.7953 ± 0.0910 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 2.0002 ± 0.0293 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.7728 ± 0.0828 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 166.8395 ± 13.5989 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 75.4283 ± 2.9032 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8297 ± 0.1242 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.5294 ± 2.5737 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7775 ± 0.1032 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6807 ± 0.0006 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 75.2536 ± 6.2615 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 48.5250 ± 1.2258 | 5 |
| structured_orth_float | off (off) | accuracy | 0.8891 ± 0.0742 | 5 |
| structured_orth_float | off (off) | loss | 1.8298 ± 0.0266 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8533 ± 0.1107 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 178.8515 ± 12.4558 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 81.9733 ± 15.8816 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8469 ± 0.0828 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 3.1598 ± 1.7239 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7810 ± 0.0821 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6807 ± 0.0006 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 87.8842 ± 16.7483 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 60.2928 ± 7.4868 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.8891 ± 0.1772 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 2.2990 ± 3.6713 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.8619 ± 0.2093 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6006 ± 0.0032 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 90.7610 ± 9.4191 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 60.1541 ± 5.0380 | 5 |

## fashion_mnist (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.8707 ± 0.0019 | 5 |
| backprop_float | off (off) | loss | 0.3659 ± 0.0027 | 5 |
| backprop_float | off (off) | macro_f1 | 0.8696 ± 0.0018 | 5 |
| backprop_float | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 97.4447 ± 2.8580 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 54.6266 ± 0.8031 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 0.8639 ± 0.0010 | 5 |
| backprop_float_lr06 | off (off) | loss | 0.3820 ± 0.0026 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.8624 ± 0.0012 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 218.8149 ± 5.2702 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 119.3485 ± 1.1880 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 0.8707 ± 0.0019 | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3659 ± 0.0027 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.8696 ± 0.0018 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 224.9068 ± 4.7572 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 120.9464 ± 1.7734 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 0.8738 ± 0.0026 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.3580 ± 0.0059 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.8729 ± 0.0025 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 221.5629 ± 2.2954 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 119.7115 ± 1.8218 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.1251 ± 0.0453 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 2.7869 ± 0.5633 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.0472 ± 0.0430 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.4795 ± 0.0188 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 90.3281 ± 28.1285 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 55.3429 ± 14.6246 | 5 |
| dfa_float | off (off) | accuracy | 0.7743 ± 0.0369 | 5 |
| dfa_float | off (off) | loss | 4.6761 ± 0.7665 | 5 |
| dfa_float | off (off) | macro_f1 | 0.7536 ± 0.0485 | 5 |
| dfa_float | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 220.9657 ± 3.4452 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 119.3608 ± 0.9764 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.8382 ± 0.0021 | 5 |
| dfa_float_clip1 | off (off) | loss | 0.4671 ± 0.0039 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.8342 ± 0.0024 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 366.3145 ± 28.6124 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 192.7326 ± 13.8219 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 0.7827 ± 0.0303 | 5 |
| dfa_float_lr06 | off (off) | loss | 4.4953 ± 0.6292 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.7695 ± 0.0403 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 222.9469 ± 4.1866 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 120.2521 ± 0.5950 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 0.7743 ± 0.0369 | 5 |
| dfa_float_lr10 | off (off) | loss | 4.6761 ± 0.7665 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.7536 ± 0.0485 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 224.0751 ± 3.2343 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 121.4028 ± 2.4288 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 0.7658 ± 0.0326 | 5 |
| dfa_float_lr15 | off (off) | loss | 4.8527 ± 0.6747 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.7476 ± 0.0461 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 223.5946 ± 10.1225 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 121.7669 ± 4.1307 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.6049 ± 0.0457 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 8.1824 ± 0.9490 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.5765 ± 0.0540 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4039 ± 0.0162 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 140.5570 ± 27.2031 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 101.3075 ± 21.6919 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.5820 ± 0.0562 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 8.6582 ± 1.1653 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.5242 ± 0.0734 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.1267 ± 0.0039 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 115.3253 ± 1.4136 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 81.8821 ± 0.3866 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.6049 ± 0.0457 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 8.1824 ± 0.9490 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.5765 ± 0.0540 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4039 ± 0.0162 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 116.4147 ± 3.0338 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 83.1030 ± 1.9774 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.6160 ± 0.0655 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 7.9532 ± 1.3517 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.5827 ± 0.0504 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.7332 ± 0.0074 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 140.4419 ± 38.4907 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 99.9529 ± 27.1149 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.2827 ± 0.0794 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 14.7691 ± 1.6928 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.1982 ± 0.0766 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6394 ± 0.0045 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 176.7348 ± 24.9422 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 111.3236 ± 14.7344 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.0999 ± 0.0040 | 5 |
| structured_hadamard_float | off (off) | loss | 15.3793 ± 7.3106 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0183 ± 0.0007 | 5 |
| structured_hadamard_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 220.8060 ± 5.6390 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 117.6346 ± 4.0111 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.7244 ± 0.0115 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.8108 ± 0.0268 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.6983 ± 0.0165 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 219.5872 ± 3.8579 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 114.5556 ± 2.1725 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.2599 ± 0.0391 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 15.2737 ± 0.8439 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.1829 ± 0.0731 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6763 ± 0.0009 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 115.1212 ± 0.7346 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 74.2931 ± 0.3922 | 5 |
| structured_orth_float | off (off) | accuracy | 0.1003 ± 0.0012 | 5 |
| structured_orth_float | off (off) | loss | 15.3683 ± 7.3039 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.0183 ± 0.0001 | 5 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 223.5788 ± 3.5405 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 118.8798 ± 1.7899 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.3705 ± 0.0421 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 12.8806 ± 0.8945 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.2991 ± 0.0515 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6704 ± 0.0011 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 152.8728 ± 33.0231 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 98.7807 ± 20.7693 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.1823 ± 0.0785 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 13.3422 ± 1.2448 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1180 ± 0.0724 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6075 ± 0.0070 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 115.4727 ± 0.9179 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 75.7955 ± 0.8391 | 5 |

## mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.3895 ± 0.0631 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 309.2660 ± 31.4422 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 149.1697 ± 8.1950 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 0.9855 ± 0.0958 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 305.0608 ± 64.3546 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 135.8510 ± 16.0469 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3895 ± 0.0631 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 304.1294 ± 42.7431 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 111.6839 ± 26.1991 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1374 ± 0.0223 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 317.5660 ± 39.5649 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 148.1184 ± 10.6000 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 5 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5215 ± 0.0346 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 168.0003 ± 13.5097 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 97.9810 ± 3.0083 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0191 ± 0.0006 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 303.1688 ± 59.4302 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 181.0149 ± 11.0666 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.2016 ± 0.0976 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2539 ± 0.0605 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1280 ± 0.0878 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 282.0742 ± 33.9770 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 157.4599 ± 31.3523 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0683 ± 0.0077 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 323.8525 ± 44.3125 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 160.7433 ± 9.9219 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0191 ± 0.0006 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 293.0458 ± 21.9257 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 160.3227 ± 23.2895 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0090 ± 0.0003 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 329.4456 ± 21.3792 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 172.7468 ± 27.5778 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6079 ± 0.0070 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 173.3292 ± 5.2660 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 123.0001 ± 8.9576 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2528 ± 0.0054 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 165.1906 ± 13.4848 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 120.7217 ± 6.5834 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6079 ± 0.0070 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 133.0554 ± 18.3925 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 107.0258 ± 23.5208 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.9187 ± 0.1178 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 1.4699 ± 2.0569 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.8943 ± 0.1447 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9501 ± 0.0013 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 157.8562 ± 19.9379 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 118.8663 ± 7.7608 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.9359 ± 0.0885 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 1.1506 ± 1.9131 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.9447 ± 0.0626 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6628 ± 0.0079 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 157.4406 ± 14.2466 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 111.8511 ± 5.1804 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9078 ± 0.0554 | 5 |
| structured_hadamard_float | off (off) | loss | 1.7804 ± 0.0639 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8764 ± 0.0705 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 302.9399 ± 18.1952 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 173.2787 ± 12.5225 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.4938 ± 0.0562 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 2.0837 ± 0.0607 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.4637 ± 0.1052 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 329.5882 ± 46.0260 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 172.0763 ± 14.6433 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8219 ± 0.0616 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.6913 ± 1.2769 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7956 ± 0.0861 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6783 ± 0.0010 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 156.4521 ± 20.5944 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 104.5589 ± 7.0718 | 5 |
| structured_orth_float | off (off) | accuracy | 0.8469 ± 0.0549 | 5 |
| structured_orth_float | off (off) | loss | 1.7855 ± 0.0700 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8245 ± 0.0866 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 289.5983 ± 23.6514 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 176.0785 ± 10.5223 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8547 ± 0.0483 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 2.9994 ± 1.0133 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7975 ± 0.0710 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6783 ± 0.0009 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 164.2910 ± 21.2824 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 115.3510 ± 4.6516 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 5 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5768 ± 0.0079 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 149.7673 ± 6.8777 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 107.4283 ± 9.3854 | 5 |

## mnist (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.9599 ± 0.0009 | 5 |
| backprop_float | off (off) | loss | 0.1355 ± 0.0017 | 5 |
| backprop_float | off (off) | macro_f1 | 0.9597 ± 0.0008 | 5 |
| backprop_float | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 245.2276 ± 13.2287 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 123.1033 ± 2.2038 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 0.9507 ± 0.0013 | 5 |
| backprop_float_lr06 | off (off) | loss | 0.1696 ± 0.0026 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.9504 ± 0.0012 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 185.3001 ± 43.9938 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 98.0524 ± 23.4906 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 0.9599 ± 0.0009 | 5 |
| backprop_float_lr10 | off (off) | loss | 0.1355 ± 0.0017 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.9597 ± 0.0008 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 195.6626 ± 37.4865 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 99.6303 ± 17.1930 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 0.9646 ± 0.0003 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1170 ± 0.0013 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.9645 ± 0.0003 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 161.6193 ± 4.1987 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 81.6983 ± 1.6221 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.3757 ± 0.1801 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 7.8465 ± 2.3446 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3352 ± 0.2059 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5536 ± 0.0110 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 118.9005 ± 11.3459 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 72.3303 ± 7.2859 | 5 |
| dfa_float | off (off) | accuracy | 0.9363 ± 0.0060 | 5 |
| dfa_float | off (off) | loss | 1.1894 ± 0.1145 | 5 |
| dfa_float | off (off) | macro_f1 | 0.9360 ± 0.0060 | 5 |
| dfa_float | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 237.5083 ± 14.5688 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 146.4357 ± 14.5010 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.9376 ± 0.0018 | 5 |
| dfa_float_clip1 | off (off) | loss | 0.2163 ± 0.0034 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.9372 ± 0.0018 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 236.6090 ± 26.8155 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 145.1826 ± 6.7638 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 0.9514 ± 0.0008 | 5 |
| dfa_float_lr06 | off (off) | loss | 0.1630 ± 0.0045 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.9511 ± 0.0008 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 189.1014 ± 27.7803 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 117.5495 ± 26.2615 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 0.9363 ± 0.0060 | 5 |
| dfa_float_lr10 | off (off) | loss | 1.1894 ± 0.1145 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.9360 ± 0.0060 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 217.6125 ± 24.1785 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 129.1371 ± 20.7967 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 0.9335 ± 0.0054 | 5 |
| dfa_float_lr15 | off (off) | loss | 1.3732 ± 0.1136 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.9332 ± 0.0054 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 236.3832 ± 36.4051 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 137.8279 ± 17.0381 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.8201 ± 0.0216 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 3.7245 ± 0.4454 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.8165 ± 0.0269 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4803 ± 0.0030 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 80.9360 ± 8.6783 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 66.9637 ± 6.4244 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.8239 ± 0.0421 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 3.6481 ± 0.8721 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.8202 ± 0.0455 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.1853 ± 0.0010 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 97.2297 ± 22.2153 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 77.5412 ± 15.2841 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.8201 ± 0.0216 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 3.7245 ± 0.4454 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.8165 ± 0.0269 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4803 ± 0.0030 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 83.0894 ± 5.3932 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 66.4458 ± 4.5753 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.7885 ± 0.0531 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.3722 ± 1.0990 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.7818 ± 0.0593 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.7480 ± 0.0035 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 100.1046 ± 25.2561 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 79.6137 ± 20.6593 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.1641 ± 0.0627 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 17.2959 ± 1.3259 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0762 ± 0.0508 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6316 ± 0.0031 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 126.8444 ± 7.6322 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 93.1741 ± 4.0420 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.7979 ± 0.0095 | 5 |
| structured_hadamard_float | off (off) | loss | 0.7630 ± 0.0282 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.7947 ± 0.0102 | 5 |
| structured_hadamard_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 162.9266 ± 4.8806 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 98.3756 ± 5.3436 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.8025 ± 0.0077 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.7444 ± 0.0227 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.7994 ± 0.0079 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 233.8413 ± 19.0208 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 137.0693 ± 9.8420 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.3566 ± 0.0380 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 13.0831 ± 0.7996 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2917 ± 0.0385 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6768 ± 0.0009 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 125.7998 ± 16.4020 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 90.9228 ± 11.2155 | 5 |
| structured_orth_float | off (off) | accuracy | 0.8074 ± 0.0074 | 5 |
| structured_orth_float | off (off) | loss | 0.7313 ± 0.0247 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8043 ± 0.0074 | 5 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 161.0175 ± 12.6589 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 97.8138 ± 5.6015 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.3603 ± 0.0360 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 13.0266 ± 0.7830 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.2933 ± 0.0438 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6768 ± 0.0009 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 82.7825 ± 4.4576 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 61.0854 ± 3.1165 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.2984 ± 0.0991 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 14.5126 ± 2.0562 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1857 ± 0.0802 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5882 ± 0.0038 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 80.8255 ± 2.7804 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 59.5446 ± 4.0772 | 5 |

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

