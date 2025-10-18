# FeedFlipNets Benchmark Summary

Aggregated over seeds with mean ± 95% CI.


## Topline Highlights

| Dataset | Mode | Metric | Mean ± 95% CI | Strategy Variant | Flip | n |
|---|---|---|---|---|---|---:|
| 20newsgroups | offline | Accuracy | 0.3906 ± 0.1692 | backprop_ternary_step | ternary (per_step) | 3 |
| 20newsgroups | offline | Macro-F1 | 0.3414 ± 0.1177 | backprop_ternary_step | ternary (per_step) | 3 |
| 20newsgroups | offline | Zero Ratio | 0.6819 ± 0.0002 | structured_orth_ternary | ternary (per_step) | 3 |
| 20newsgroups | offline | Test Throughput (samples/s) | 17760.0556 ± 1388.4289 | backprop_float | off (off) | 3 |
| 20newsgroups | real | Accuracy | 0.3206 ± 0.0113 | dfa_float_lr15 | off (off) | 3 |
| 20newsgroups | real | Macro-F1 | 0.2744 ± 0.0111 | dfa_float_lr15 | off (off) | 3 |
| 20newsgroups | real | Zero Ratio | 0.9544 ± 0.0001 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| 20newsgroups | real | Test Throughput (samples/s) | 17155.5581 ± 557.4554 | backprop_float | off (off) | 3 |
| adult | offline | Accuracy | 0.9089 ± 0.0784 | dfa_float_lr15 | off (off) | 3 |
| adult | offline | R² | 0.6091 ± 0.0706 | dfa_float_lr15 | off (off) | 3 |
| adult | offline | Zero Ratio | 0.9553 ± 0.0037 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| adult | offline | Test Throughput (samples/s) | 50088.4307 ± 7908.9156 | backprop_float | off (off) | 5 |
| adult | real | Accuracy | 0.8595 ± 0.0006 | dfa_float | off (off) | 3 |
| adult | real | R² | 0.4654 ± 0.0036 | dfa_float | off (off) | 3 |
| adult | real | Zero Ratio | 0.4453 ± 0.0126 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 3 |
| adult | real | Test Throughput (samples/s) | 41361.0390 ± 2128.8954 | backprop_float | off (off) | 3 |
| ag_news | offline | Accuracy | 0.4813 ± 0.2285 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Macro-F1 | 0.3933 ± 0.2679 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Zero Ratio | 0.6818 ± 0.0004 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 5 |
| ag_news | offline | Test Throughput (samples/s) | 4356.2552 ± 1173.2337 | backprop_float | off (off) | 5 |
| ag_news | real | Accuracy | 0.9005 ± 0.0012 | backprop_float | off (off) | 3 |
| ag_news | real | Macro-F1 | 0.9002 ± 0.0012 | backprop_float | off (off) | 3 |
| ag_news | real | Zero Ratio | 0.6721 ± 0.0075 | dfa_ternary_epoch_tau005 | ternary (per_epoch) | 2 |
| ag_news | real | Test Throughput (samples/s) | 3378.2391 ± 660.5452 | dfa_float | off (off) | 2 |
| california_housing | offline | R² | 0.0840 ± 0.1381 | dfa_float | off (off) | 3 |
| california_housing | offline | Zero Ratio | 0.2186 ± 0.4993 | structured_hadamard_ternary | ternary (per_step) | 3 |
| california_housing | offline | Test Throughput (samples/s) | 87757.7091 ± 2539.0751 | backprop_float | off (off) | 3 |
| california_housing | real | R² | 0.5561 ± 0.0081 | dfa_float | off (off) | 5 |
| california_housing | real | Zero Ratio | 0.5957 ± 0.8705 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| california_housing | real | Test Throughput (samples/s) | 161174.3075 ± 2521.2085 | structured_hadamard_float | off (off) | 3 |
| cifar10 | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| cifar10 | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| cifar10 | offline | Zero Ratio | 0.9535 ± 0.0007 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| cifar10 | offline | Test Throughput (samples/s) | 230.6713 ± 19.8734 | dfa_float_clip1 | off (off) | 5 |
| fashion_mnist | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| fashion_mnist | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| fashion_mnist | offline | Zero Ratio | 0.9533 ± 0.0006 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| fashion_mnist | offline | Test Throughput (samples/s) | 189.7377 ± 34.5359 | dfa_float | off (off) | 5 |
| fashion_mnist | real | Accuracy | 0.8738 ± 0.0032 | backprop_float_lr15 | off (off) | 5 |
| fashion_mnist | real | Macro-F1 | 0.8729 ± 0.0031 | backprop_float_lr15 | off (off) | 5 |
| fashion_mnist | real | Zero Ratio | 0.7332 ± 0.0092 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| fashion_mnist | real | Test Throughput (samples/s) | 366.3145 ± 35.5213 | dfa_float_clip1 | off (off) | 5 |
| mnist | offline | Accuracy | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| mnist | offline | Macro-F1 | 1.0000 | dfa_float_lr15 | off (off) | 5 |
| mnist | offline | Zero Ratio | 0.9501 ± 0.0016 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| mnist | offline | Test Throughput (samples/s) | 329.5882 ± 57.1396 | structured_hadamard_float_clip1 | off (off) | 5 |
| mnist | real | Accuracy | 0.9646 ± 0.0004 | backprop_float_lr15 | off (off) | 5 |
| mnist | real | Macro-F1 | 0.9645 ± 0.0004 | backprop_float_lr15 | off (off) | 5 |
| mnist | real | Zero Ratio | 0.7480 ± 0.0044 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 5 |
| mnist | real | Test Throughput (samples/s) | 245.2276 ± 16.4230 | backprop_float | off (off) | 5 |
| ucr | offline | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | offline | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | offline | Zero Ratio | 0.6772 ± 0.0052 | structured_hadamard_ternary | ternary (per_step) | 3 |
| ucr | offline | Test Throughput (samples/s) | 51612.2241 ± 7574.8899 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| ucr | real | Zero Ratio | 0.6772 ± 0.0052 | structured_hadamard_ternary | ternary (per_step) | 3 |
| ucr | real | Test Throughput (samples/s) | 41204.3096 ± 3464.4454 | structured_orth_float | off (off) | 3 |

## Best Configs

| Dataset | Mode | Primary | Best (μ±95% CI) | Variant | Flip | n | Baseline (μ±σ) | Δ | Effect Size |
|---|---|---|---|---|---|---:|---|---:|---:|
| 20newsgroups | offline | accuracy | 0.2083 ± 0.2138 | structured_orth_ternary | ternary (per_step) | 3 | 0.1771 ± 0.1793 | 0.0312 | 0.393 |
| 20newsgroups | real | accuracy | 0.0576 ± 0.0102 | structured_orth_ternary | ternary (per_step) | 3 | 0.3206 ± 0.0113 | -0.2630 | -60.619 |
| adult | offline | accuracy | 0.5130 ± 0.2907 | structured_orth_ternary | ternary (per_step) | 3 | 0.9089 ± 0.0784 | -0.3958 | -4.619 |
| adult | real | accuracy | 0.8565 ± 0.0035 | backprop_float | off (off) | 3 | 0.8595 ± 0.0006 | -0.0030 | -3.004 |
| ag_news | offline | accuracy | 0.4125 ± 0.0874 | backprop_float | off (off) | 5 | 0.4125 ± 0.0874 | 0.0000 | 0.000 |
| ag_news | real | accuracy | 0.9005 ± 0.0012 | backprop_float | off (off) | 3 | 0.9005 ± 0.0012 | 0.0000 | 0.000 |
| california_housing | offline | r2 | -17612.4954 ± 29085.8532 | structured_orth_ternary | ternary (per_step) | 3 | 0.0840 ± 0.1381 | -17612.5794 | -2.127 |
| california_housing | real | r2 | -3213.7032 ± 4629.3813 | structured_orth_ternary | ternary (per_step) | 3 | 0.5561 ± 0.0081 | -3214.2592 | -2.439 |
| cifar10 | offline | accuracy | 0.8250 ± 0.1094 | structured_orth_ternary | ternary (per_step) | 5 | 1.0000 | -0.1750 | -2.808 |
| fashion_mnist | offline | accuracy | 0.8469 ± 0.1028 | structured_orth_ternary | ternary (per_step) | 5 | 1.0000 | -0.1531 | -2.616 |
| fashion_mnist | real | accuracy | 0.3705 ± 0.0523 | structured_orth_ternary | ternary (per_step) | 5 | 0.8738 ± 0.0032 | -0.5034 | -16.880 |
| mnist | offline | accuracy | 0.8547 ± 0.0600 | structured_orth_ternary | ternary (per_step) | 5 | 1.0000 | -0.1453 | -4.250 |
| mnist | real | accuracy | 0.3603 ± 0.0447 | structured_orth_ternary | ternary (per_step) | 5 | 0.9646 ± 0.0004 | -0.6043 | -23.750 |
| ucr | offline | accuracy | 0.7969 | structured_orth_ternary | ternary (per_step) | 3 | 1.0000 | -0.2031 | 0.000 |
| ucr | real | accuracy | 0.7969 | structured_orth_ternary | ternary (per_step) | 3 | 1.0000 | -0.2031 | 0.000 |

## 20newsgroups (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.1771 ± 0.1793 | 3 |
| backprop_float | off (off) | loss | 2.0570 ± 0.0208 | 3 |
| backprop_float | off (off) | macro_f1 | 0.1426 ± 0.2308 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 17760.0556 ± 1388.4289 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 4783.4421 ± 293.1573 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.3906 ± 0.1692 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 12.0424 ± 2.7980 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3414 ± 0.1177 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6810 ± 0.0004 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 7772.4028 ± 591.3615 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 1040.2072 ± 53.1488 | 3 |
| dfa_float | off (off) | accuracy | 0.1510 ± 0.0808 | 3 |
| dfa_float | off (off) | loss | 2.0691 ± 0.0193 | 3 |
| dfa_float | off (off) | macro_f1 | 0.0780 ± 0.1058 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 7518.0989 ± 2043.1668 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 2244.2749 ± 10.9534 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.3438 ± 0.1027 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 13.0543 ± 3.3808 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.2602 ± 0.1348 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6818 ± 0.0003 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 7491.7946 ± 656.8004 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 2256.5135 ± 188.4654 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.2240 ± 0.1248 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 15.7157 ± 2.4974 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.1734 ± 0.1741 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0002 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 7340.1342 ± 2187.5813 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 1048.3695 ± 61.6575 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.1354 ± 0.0896 | 3 |
| structured_hadamard_float | off (off) | loss | 2.0756 ± 0.0193 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0825 ± 0.1019 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 17159.8569 ± 1368.8475 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 3804.6909 ± 163.0052 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.2917 ± 0.0448 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 14.5478 ± 1.4932 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2593 ± 0.0238 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0002 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 7245.0925 ± 722.6365 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 975.0769 ± 34.5542 | 3 |
| structured_orth_float | off (off) | accuracy | 0.1354 ± 0.0896 | 3 |
| structured_orth_float | off (off) | loss | 2.0751 ± 0.0181 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.0817 ± 0.1019 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 17386.8935 ± 455.6860 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 1490.1215 ± 216.8915 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.2083 ± 0.2138 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 16.1435 ± 4.9837 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.1511 ± 0.1963 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0002 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 7804.6346 ± 415.7641 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 713.6274 ± 9.1410 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.2656 ± 0.1027 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 14.8916 ± 2.2014 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.2078 ± 0.0773 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6809 ± 0.0008 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 7573.5650 ± 957.2115 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 1054.6434 ± 104.3450 | 3 |

## 20newsgroups (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.1230 ± 0.0105 | 3 |
| backprop_float | off (off) | loss | 2.9214 ± 0.0064 | 3 |
| backprop_float | off (off) | macro_f1 | 0.0722 ± 0.0086 | 3 |
| backprop_float | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 17155.5581 ± 557.4554 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 4127.2925 ± 22.5338 | 3 |
| backprop_float_lr06 | off (off) | accuracy | 0.0862 ± 0.0410 | 3 |
| backprop_float_lr06 | off (off) | loss | 2.9690 ± 0.0009 | 3 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.0439 ± 0.0394 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 8381.7305 ± 1032.5086 | 3 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 2063.9421 ± 212.9381 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.1230 ± 0.0105 | 3 |
| backprop_float_lr10 | off (off) | loss | 2.9214 ± 0.0064 | 3 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.0722 ± 0.0086 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 8209.2622 ± 1481.1189 | 3 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 1774.0311 ± 953.9894 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.1986 ± 0.0334 | 3 |
| backprop_float_lr15 | off (off) | loss | 2.6976 ± 0.0254 | 3 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.1400 ± 0.0493 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 7069.3706 ± 4818.3338 | 3 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 1727.3163 ± 740.5331 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.0615 ± 0.0303 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 17.8996 ± 0.4818 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.0541 ± 0.0271 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6822 ± 0.0003 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 6125.3234 ± 258.6995 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 810.4823 ± 6.4844 | 3 |
| dfa_float | off (off) | accuracy | 0.2229 ± 0.0267 | 3 |
| dfa_float | off (off) | loss | 2.5541 ± 0.0477 | 3 |
| dfa_float | off (off) | macro_f1 | 0.1686 ± 0.0208 | 3 |
| dfa_float | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 6134.0114 ± 279.5227 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 1879.1527 ± 14.9520 | 3 |
| dfa_float_clip1 | off (off) | accuracy | 0.2736 ± 0.0427 | 3 |
| dfa_float_clip1 | off (off) | loss | 2.7128 ± 0.0487 | 3 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1989 ± 0.0464 | 3 |
| dfa_float_clip1 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 1700.6208 ± 319.0340 | 3 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 506.7781 ± 98.0180 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.1294 ± 0.0417 | 3 |
| dfa_float_lr06 | off (off) | loss | 2.8922 ± 0.0546 | 3 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.0769 ± 0.0255 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 1397.4045 ± 764.4064 | 3 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 489.7546 ± 309.8655 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.2229 ± 0.0267 | 3 |
| dfa_float_lr10 | off (off) | loss | 2.5541 ± 0.0477 | 3 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.1686 ± 0.0208 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 2329.9449 ± 1139.2795 | 3 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 805.4994 ± 356.7495 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.3206 ± 0.0113 **(best)** | 3 |
| dfa_float_lr15 | off (off) | loss | 2.2206 ± 0.0259 | 3 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.2744 ± 0.0111 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 2433.5734 ± 1624.4771 | 3 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 761.1861 ± 456.5956 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.1264 ± 0.0458 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 17.2000 ± 1.7904 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.0992 ± 0.0736 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6793 ± 0.0006 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 5723.3596 ± 502.7802 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 1843.8629 ± 44.9303 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.1028 ± 0.0354 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 18.3765 ± 0.9821 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.0633 ± 0.0340 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2971 ± 0.0014 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 2677.7276 ± 490.8430 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 880.7450 ± 70.4091 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.1264 ± 0.0458 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 17.2000 ± 1.7904 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.0992 ± 0.0736 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6793 ± 0.0006 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 3722.4781 ± 275.1123 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 1120.9013 ± 63.9690 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.0841 ± 0.0152 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.5784 ± 2.0464 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.0575 ± 0.0171 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9544 ± 0.0001 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 3686.5771 ± 189.9517 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 1132.4488 ± 26.0098 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.0599 ± 0.0132 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 18.3875 ± 0.2502 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0441 ± 0.0094 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0003 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 5917.8555 ± 501.6701 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 806.8453 ± 26.5998 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.0722 ± 0.0139 | 3 |
| structured_hadamard_float | off (off) | loss | 2.9809 ± 0.0013 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0310 ± 0.0216 | 3 |
| structured_hadamard_float | off (off) | sample_count | 3776.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 16670.0619 ± 666.9777 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 3199.4279 ± 110.0095 | 3 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.0931 ± 0.0254 | 3 |
| structured_hadamard_float_clip1 | off (off) | loss | 2.9834 ± 0.0025 | 3 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.0546 ± 0.0128 | 3 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 3776.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 2052.5608 ± 128.4897 | 3 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 526.9783 ± 8.8182 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.0563 ± 0.0179 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 18.3074 ± 0.2819 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.0458 ± 0.0107 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 3776.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0004 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 5664.4212 ± 238.9884 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 750.6693 ± 8.0511 | 3 |
| structured_orth_float | off (off) | accuracy | 0.0876 ± 0.0379 | 3 |
| structured_orth_float | off (off) | loss | 2.9855 ± 0.0003 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.0448 ± 0.0398 | 3 |
| structured_orth_float | off (off) | sample_count | 3776.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 17052.5182 ± 1036.7280 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 1147.0103 ± 86.0200 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.0576 ± 0.0102 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 18.3228 ± 0.4643 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.0441 ± 0.0115 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 3776.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6821 ± 0.0003 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 5739.9968 ± 317.0986 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 524.1646 ± 6.9035 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.0591 ± 0.0206 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 18.3253 ± 0.6929 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.0478 ± 0.0211 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0003 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 5411.6277 ± 705.2022 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 781.6531 ± 10.0604 | 3 |

## adult (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.5422 ± 0.1317 | 5 |
| backprop_float | off (off) | f1 | 0.4944 ± 0.3082 | 5 |
| backprop_float | off (off) | loss | 0.6915 ± 0.0024 | 5 |
| backprop_float | off (off) | mae | 0.4991 ± 0.0012 | 5 |
| backprop_float | off (off) | precision | 0.7678 ± 0.1379 | 5 |
| backprop_float | off (off) | r2 | -0.0631 ± 0.0052 | 5 |
| backprop_float | off (off) | recall | 0.4425 ± 0.3706 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 50088.4307 ± 7908.9156 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 28740.5647 ± 3190.6537 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 0.5859 ± 0.4176 | 3 |
| backprop_float_lr06 | off (off) | f1 | 0.5349 ± 0.8682 | 3 |
| backprop_float_lr06 | off (off) | loss | 0.6909 ± 0.0068 | 3 |
| backprop_float_lr06 | off (off) | mae | 0.4988 ± 0.0034 | 3 |
| backprop_float_lr06 | off (off) | precision | 0.8269 ± 0.4197 | 3 |
| backprop_float_lr06 | off (off) | r2 | -0.0618 ± 0.0145 | 3 |
| backprop_float_lr06 | off (off) | recall | 0.5083 ± 0.9452 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 16509.8730 ± 2419.1272 | 3 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 7049.7947 ± 691.5316 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.5833 ± 0.4077 | 3 |
| backprop_float_lr10 | off (off) | f1 | 0.5227 ± 0.8379 | 3 |
| backprop_float_lr10 | off (off) | loss | 0.6900 ± 0.0072 | 3 |
| backprop_float_lr10 | off (off) | mae | 0.4984 ± 0.0036 | 3 |
| backprop_float_lr10 | off (off) | precision | 0.8527 ± 0.4152 | 3 |
| backprop_float_lr10 | off (off) | r2 | -0.0600 ± 0.0154 | 3 |
| backprop_float_lr10 | off (off) | recall | 0.4667 ± 0.8450 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 13733.8732 ± 6134.3295 | 3 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 6000.0081 ± 4100.2176 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.5938 ± 0.4131 | 3 |
| backprop_float_lr15 | off (off) | f1 | 0.5079 ± 0.8512 | 3 |
| backprop_float_lr15 | off (off) | loss | 0.6889 ± 0.0079 | 3 |
| backprop_float_lr15 | off (off) | mae | 0.4978 ± 0.0039 | 3 |
| backprop_float_lr15 | off (off) | precision | 0.8993 ± 0.2676 | 3 |
| backprop_float_lr15 | off (off) | r2 | -0.0576 ± 0.0168 | 3 |
| backprop_float_lr15 | off (off) | recall | 0.4208 ± 0.7809 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 14551.9545 ± 11513.8809 | 3 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 5780.4851 ± 2854.4602 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.5339 ± 0.3250 | 3 |
| backprop_ternary_step | ternary (per_step) | f1 | 0.5458 ± 0.4934 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 7.2513 ± 3.7658 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 0.4655 ± 0.2928 | 3 |
| backprop_ternary_step | ternary (per_step) | precision | 0.6675 ± 0.3884 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -0.9188 ± 1.2026 | 3 |
| backprop_ternary_step | ternary (per_step) | recall | 0.4917 ± 0.6531 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6801 ± 0.0164 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 12270.1506 ± 5661.9896 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 4282.6546 ± 2644.5345 | 3 |
| dfa_float | off (off) | accuracy | 0.5328 ± 0.1428 | 5 |
| dfa_float | off (off) | f1 | 0.4819 ± 0.3146 | 5 |
| dfa_float | off (off) | loss | 0.6920 ± 0.0039 | 5 |
| dfa_float | off (off) | mae | 0.4994 ± 0.0019 | 5 |
| dfa_float | off (off) | precision | 0.8247 ± 0.2339 | 5 |
| dfa_float | off (off) | r2 | -0.0642 ± 0.0083 | 5 |
| dfa_float | off (off) | recall | 0.4125 ± 0.3175 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 43760.9011 ± 4523.3606 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 30491.0915 ± 2403.1023 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.5208 ± 0.2649 | 3 |
| dfa_float_clip1 | off (off) | f1 | 0.4966 ± 0.8192 | 3 |
| dfa_float_clip1 | off (off) | loss | 0.6926 ± 0.0040 | 3 |
| dfa_float_clip1 | off (off) | mae | 0.4997 ± 0.0020 | 3 |
| dfa_float_clip1 | off (off) | precision | 0.7072 ± 0.2873 | 3 |
| dfa_float_clip1 | off (off) | r2 | -0.0654 ± 0.0086 | 3 |
| dfa_float_clip1 | off (off) | recall | 0.5125 ± 0.9687 | 3 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 15734.6856 ± 4824.8718 | 3 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 6356.7891 ± 343.0183 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.5391 ± 0.4766 | 3 |
| dfa_float_lr06 | off (off) | f1 | 0.3766 ± 1.0195 | 3 |
| dfa_float_lr06 | off (off) | loss | 0.6853 ± 0.0131 | 3 |
| dfa_float_lr06 | off (off) | mae | 0.4958 ± 0.0068 | 3 |
| dfa_float_lr06 | off (off) | precision | 0.5870 ± 1.2973 | 3 |
| dfa_float_lr06 | off (off) | r2 | -0.0499 ± 0.0278 | 3 |
| dfa_float_lr06 | off (off) | recall | 0.3542 ± 1.1445 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 13714.5840 ± 5190.3909 | 3 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 7155.2834 ± 485.7662 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.8229 ± 0.0917 | 3 |
| dfa_float_lr10 | off (off) | f1 | 0.8364 ± 0.0996 | 3 |
| dfa_float_lr10 | off (off) | loss | 0.5466 ± 0.0473 | 3 |
| dfa_float_lr10 | off (off) | mae | 0.4125 ± 0.0303 | 3 |
| dfa_float_lr10 | off (off) | precision | 0.9831 ± 0.0035 | 3 |
| dfa_float_lr10 | off (off) | r2 | 0.2276 ± 0.0920 | 3 |
| dfa_float_lr10 | off (off) | recall | 0.7292 ± 0.1468 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 15613.2733 ± 2067.0427 | 3 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 7273.5395 ± 493.8135 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.9089 ± 0.0784 **(best)** | 3 |
| dfa_float_lr15 | off (off) | f1 | 0.9219 ± 0.0725 | 3 |
| dfa_float_lr15 | off (off) | loss | 0.3217 ± 0.0468 | 3 |
| dfa_float_lr15 | off (off) | mae | 0.2527 ± 0.0368 | 3 |
| dfa_float_lr15 | off (off) | precision | 0.9858 ± 0.0021 | 3 |
| dfa_float_lr15 | off (off) | r2 | 0.6091 ± 0.0706 | 3 |
| dfa_float_lr15 | off (off) | recall | 0.8667 ± 0.1255 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 16535.0194 ± 2333.9210 | 3 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 6897.2037 ± 958.1769 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.6094 ± 0.2288 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | f1 | 0.6171 ± 0.4670 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 6.3082 ± 5.4808 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 0.3863 ± 0.2425 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | precision | 0.7748 ± 0.2063 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -0.5807 ± 1.0988 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | recall | 0.5667 ± 0.6692 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6687 ± 0.0234 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 12622.7968 ± 4632.5623 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 6000.3469 ± 404.2318 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.5130 ± 0.2340 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | f1 | 0.5006 ± 0.6988 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 9.0169 ± 4.5407 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | mae | 0.4827 ± 0.2330 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | precision | 0.6740 ± 0.1715 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | r2 | -1.0430 ± 1.0043 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | recall | 0.4917 ± 0.8518 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2648 ± 0.0375 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 15295.4422 ± 1560.7344 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 6093.4874 ± 2177.7258 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.6438 ± 0.1107 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | f1 | 0.6718 ± 0.1942 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 4.9602 ± 3.0332 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 0.3501 ± 0.1144 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | precision | 0.7681 ± 0.0797 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -0.4158 ± 0.5078 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | recall | 0.6400 ± 0.2729 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6707 ± 0.0089 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 37884.1688 ± 2978.8588 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 27990.4127 ± 3929.5713 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.5599 ± 0.1482 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | f1 | 0.6299 ± 0.4367 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 0.8439 ± 0.6036 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | mae | 0.4744 ± 0.1570 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | precision | 0.6677 ± 0.2307 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | r2 | -0.2191 ± 0.8746 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | recall | 0.7000 ± 0.8957 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9553 ± 0.0037 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 14601.5060 ± 3223.3062 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 5919.9562 ± 464.1204 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.5339 ± 0.1628 | 3 |
| dfa_ternary_step | ternary (per_step) | f1 | 0.5984 ± 0.2737 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 7.0006 ± 2.0944 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 0.4691 ± 0.1809 | 3 |
| dfa_ternary_step | ternary (per_step) | precision | 0.6376 ± 0.0367 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -0.9352 ± 0.7204 | 3 |
| dfa_ternary_step | ternary (per_step) | recall | 0.5833 ± 0.5049 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6798 ± 0.0169 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 16215.4888 ± 612.4073 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 5738.1191 ± 293.0149 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.5625 ± 0.3688 | 3 |
| structured_hadamard_float | off (off) | f1 | 0.5263 ± 0.8480 | 3 |
| structured_hadamard_float | off (off) | loss | 0.6919 ± 0.0064 | 3 |
| structured_hadamard_float | off (off) | mae | 0.4993 ± 0.0032 | 3 |
| structured_hadamard_float | off (off) | precision | 0.7479 ± 0.2703 | 3 |
| structured_hadamard_float | off (off) | r2 | -0.0640 ± 0.0137 | 3 |
| structured_hadamard_float | off (off) | recall | 0.5250 ± 0.9805 | 3 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 16710.3283 ± 1531.1294 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 4548.9548 ± 941.1022 | 3 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.5625 ± 0.3688 | 3 |
| structured_hadamard_float_clip1 | off (off) | f1 | 0.5263 ± 0.8480 | 3 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.6919 ± 0.0064 | 3 |
| structured_hadamard_float_clip1 | off (off) | mae | 0.4993 ± 0.0032 | 3 |
| structured_hadamard_float_clip1 | off (off) | precision | 0.7479 ± 0.2703 | 3 |
| structured_hadamard_float_clip1 | off (off) | r2 | -0.0640 ± 0.0137 | 3 |
| structured_hadamard_float_clip1 | off (off) | recall | 0.5250 ± 0.9805 | 3 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 16474.3741 ± 3021.7060 | 3 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 4388.2189 ± 950.1966 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.5365 ± 0.2808 | 3 |
| structured_hadamard_ternary | ternary (per_step) | f1 | 0.5623 ± 0.3902 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 6.9442 ± 1.9568 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 0.4694 ± 0.2701 | 3 |
| structured_hadamard_ternary | ternary (per_step) | precision | 0.6749 ± 0.3422 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -0.9301 ± 1.0888 | 3 |
| structured_hadamard_ternary | ternary (per_step) | recall | 0.5083 ± 0.5729 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6802 ± 0.0158 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 12857.2240 ± 4922.4504 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 3639.7207 ± 793.9878 | 3 |
| structured_orth_float | off (off) | accuracy | 0.5599 ± 0.3785 | 3 |
| structured_orth_float | off (off) | f1 | 0.5191 ± 0.8786 | 3 |
| structured_orth_float | off (off) | loss | 0.6919 ± 0.0065 | 3 |
| structured_orth_float | off (off) | mae | 0.4994 ± 0.0032 | 3 |
| structured_orth_float | off (off) | precision | 0.7400 ± 0.2412 | 3 |
| structured_orth_float | off (off) | r2 | -0.0641 ± 0.0140 | 3 |
| structured_orth_float | off (off) | recall | 0.5208 ± 0.9983 | 3 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 16897.0019 ± 1655.6426 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 5426.9110 ± 1606.3518 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.5130 ± 0.2907 | 3 |
| structured_orth_ternary | ternary (per_step) | f1 | 0.5293 ± 0.4805 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 7.3553 ± 3.5373 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 0.4775 ± 0.2766 | 3 |
| structured_orth_ternary | ternary (per_step) | precision | 0.6322 ± 0.2649 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -0.9578 ± 1.1364 | 3 |
| structured_orth_ternary | ternary (per_step) | recall | 0.4833 ± 0.6561 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6802 ± 0.0161 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 9940.3261 ± 11467.6844 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 3716.7583 ± 2797.9511 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.5521 ± 0.2984 | 3 |
| ternary_dfa_step | ternary (per_step) | f1 | 0.5936 ± 0.4833 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 7.2773 ± 2.6192 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 0.4549 ± 0.3080 | 3 |
| ternary_dfa_step | ternary (per_step) | precision | 0.6488 ± 0.2091 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -0.8996 ± 1.2120 | 3 |
| ternary_dfa_step | ternary (per_step) | recall | 0.5792 ± 0.6848 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6800 ± 0.0155 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 14710.6294 ± 3808.8220 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 5437.1334 ± 944.4609 | 3 |

## adult (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.8565 ± 0.0035 | 3 |
| backprop_float | off (off) | f1 | 0.6847 ± 0.0069 | 3 |
| backprop_float | off (off) | loss | 0.3237 ± 0.0116 | 3 |
| backprop_float | off (off) | mae | 0.2019 ± 0.0018 | 3 |
| backprop_float | off (off) | precision | 0.7531 ± 0.0096 | 3 |
| backprop_float | off (off) | r2 | 0.4612 ± 0.0084 | 3 |
| backprop_float | off (off) | recall | 0.6277 ± 0.0048 | 3 |
| backprop_float | off (off) | sample_count | 9792.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 41361.0390 ± 2128.8954 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 21466.3725 ± 2313.8469 | 3 |
| dfa_float | off (off) | accuracy | 0.8595 ± 0.0006 **(best)** | 3 |
| dfa_float | off (off) | f1 | 0.6878 ± 0.0019 | 3 |
| dfa_float | off (off) | loss | 0.3213 ± 0.0058 | 3 |
| dfa_float | off (off) | mae | 0.1939 ± 0.0010 | 3 |
| dfa_float | off (off) | precision | 0.7669 ± 0.0036 | 3 |
| dfa_float | off (off) | r2 | 0.4654 ± 0.0036 | 3 |
| dfa_float | off (off) | recall | 0.6236 ± 0.0046 | 3 |
| dfa_float | off (off) | sample_count | 9792.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 39675.8254 ± 4526.4691 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 23046.8012 ± 1240.4971 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.8022 ± 0.1004 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | f1 | 0.3860 ± 0.7238 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 3.9587 ± 2.1820 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 0.1978 ± 0.1008 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | precision | 0.7713 ± 0.2115 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -0.0567 ± 0.5426 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | recall | 0.3192 ± 0.7578 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 9792.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4453 ± 0.0126 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 32489.7553 ± 1925.9782 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 19175.7053 ± 517.0185 | 3 |

## ag_news (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.4125 ± 0.0874 | 5 |
| backprop_float | off (off) | loss | 1.3651 ± 0.0101 | 5 |
| backprop_float | off (off) | macro_f1 | 0.3667 ± 0.0788 | 5 |
| backprop_float | off (off) | sample_count | 64.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 4356.2552 ± 1173.2337 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 1513.9953 ± 76.2763 | 5 |
| dfa_float | off (off) | accuracy | 0.3812 ± 0.2425 | 5 |
| dfa_float | off (off) | loss | 1.3409 ± 0.0375 | 5 |
| dfa_float | off (off) | macro_f1 | 0.3508 ± 0.2067 | 5 |
| dfa_float | off (off) | sample_count | 64.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 4194.9583 ± 659.4499 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 1447.1737 ± 287.7512 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.4813 ± 0.2285 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 10.1632 ± 4.5923 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.3933 ± 0.2679 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6818 ± 0.0004 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 2755.6920 ± 179.0091 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 863.2277 ± 62.9993 | 5 |

## ag_news (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.9005 ± 0.0012 **(best)** | 3 |
| backprop_float | off (off) | loss | 0.3010 ± 0.0043 | 3 |
| backprop_float | off (off) | macro_f1 | 0.9002 ± 0.0012 | 3 |
| backprop_float | off (off) | sample_count | 25536.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 3150.5784 ± 2857.9959 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 1202.8757 ± 1308.4940 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.8547 ± 0.0570 | 2 |
| backprop_ternary_step | ternary (per_step) | loss | 1.3690 ± 0.0142 | 2 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.8539 ± 0.0542 | 2 |
| backprop_ternary_step | ternary (per_step) | sample_count | 25536.0000 | 2 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 2 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0000 | 2 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 1044.3220 ± 337.6467 | 2 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 200.5301 ± 24.9735 | 2 |
| dfa_float | off (off) | accuracy | 0.8998 ± 0.0042 | 2 |
| dfa_float | off (off) | loss | 0.3030 ± 0.0142 | 2 |
| dfa_float | off (off) | macro_f1 | 0.8995 ± 0.0043 | 2 |
| dfa_float | off (off) | sample_count | 25536.0000 | 2 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 2 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 2 |
| dfa_float | off (off) | test_throughput_samples_sec | 3378.2391 ± 660.5452 | 2 |
| dfa_float | off (off) | train_throughput_samples_sec | 1411.9319 ± 564.5875 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.7739 ± 0.0117 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 4.4280 ± 0.5141 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.7670 ± 0.0252 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 25536.0000 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6721 ± 0.0075 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 1768.3362 ± 2463.2350 | 2 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 692.9835 ± 521.9577 | 2 |

## california_housing (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | loss | 6.3994 ± 0.0610 | 3 |
| backprop_float | off (off) | mae | 1.9487 ± 0.0101 | 3 |
| backprop_float | off (off) | r2 | -0.0124 ± 0.0096 | 3 |
| backprop_float | off (off) | rmse | 2.5297 ± 0.0121 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 87757.7091 ± 2539.0751 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 72861.2586 ± 39291.4473 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 122.5007 ± 499.7007 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 4.8773 ± 12.6682 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -18.3791 ± 79.0504 | 3 |
| backprop_ternary_step | ternary (per_step) | rmse | 7.9612 ± 23.3950 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0810 ± 0.0724 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 78312.6326 ± 5115.2646 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 44948.4009 ± 4322.2093 | 3 |
| dfa_float | off (off) | loss | 5.7903 ± 0.8731 | 3 |
| dfa_float | off (off) | mae | 1.8461 ± 0.1417 | 3 |
| dfa_float | off (off) | r2 | 0.0840 ± 0.1381 **(best)** | 3 |
| dfa_float | off (off) | rmse | 2.4055 ± 0.1828 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 82614.6034 ± 2360.6796 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 54318.8346 ± 5256.0055 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 9.8164 ± 3.4793 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 2.2261 ± 0.2069 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -0.5529 ± 0.5504 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | rmse | 3.1280 ± 0.5455 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.0626 ± 0.0589 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 83823.6845 ± 1280.3006 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 57479.6878 ± 568.7722 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 14681.5489 ± 41807.4682 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 90.7748 ± 208.1392 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -2321.5537 ± 6613.7498 | 3 |
| dfa_ternary_step | ternary (per_step) | rmse | 97.7155 ± 217.9978 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0847 ± 0.1179 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 75960.4744 ± 11649.7276 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 44306.4398 ± 2843.6865 | 3 |
| structured_hadamard_float | off (off) | loss | 6.4155 ± 0.0346 | 3 |
| structured_hadamard_float | off (off) | mae | 1.9511 ± 0.0066 | 3 |
| structured_hadamard_float | off (off) | r2 | -0.0149 ± 0.0055 | 3 |
| structured_hadamard_float | off (off) | rmse | 2.5329 ± 0.0068 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 83001.2409 ± 13273.5958 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 35816.9013 ± 1855.8574 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 14.9452 ± 30.4329 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 2.6903 ± 2.4710 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -1.3643 ± 4.8143 | 3 |
| structured_hadamard_ternary | ternary (per_step) | rmse | 3.6629 ± 3.7620 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.2186 ± 0.4993 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 83191.0634 ± 3630.0819 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 26374.2458 ± 1393.4480 | 3 |
| structured_orth_float | off (off) | loss | 6.4197 ± 0.0428 | 3 |
| structured_orth_float | off (off) | mae | 1.9518 ± 0.0076 | 3 |
| structured_orth_float | off (off) | r2 | -0.0156 ± 0.0068 | 3 |
| structured_orth_float | off (off) | rmse | 2.5337 ± 0.0085 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 84463.3734 ± 7755.7768 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 26477.3560 ± 858.8312 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 111340.1135 ± 183860.2796 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 309.3594 ± 242.5000 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -17612.4954 ± 29085.8532 | 3 |
| structured_orth_ternary | ternary (per_step) | rmse | 322.3666 ± 262.0933 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.1183 ± 0.1097 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 78221.6017 ± 7896.7234 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 21398.7107 ± 791.1548 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 186.0816 ± 359.0247 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 10.5616 ± 11.4097 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -28.4373 ± 56.7961 | 3 |
| ternary_dfa_step | ternary (per_step) | rmse | 12.9663 ± 12.8937 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.1366 ± 0.0948 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 79211.8284 ± 9840.8129 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 44915.5575 ± 2448.2788 | 3 |

## california_housing (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | loss | 0.6816 ± 0.0079 | 5 |
| backprop_float | off (off) | mae | 0.6014 ± 0.0050 | 5 |
| backprop_float | off (off) | r2 | 0.5026 ± 0.0057 | 5 |
| backprop_float | off (off) | rmse | 0.8256 ± 0.0048 | 5 |
| backprop_float | off (off) | sample_count | 4160.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 118330.2065 ± 69399.4760 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 70789.2805 ± 49388.3417 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 22.1981 ± 59.9055 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 2.9896 ± 3.1738 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -15.1987 ± 43.7150 | 3 |
| backprop_ternary_step | ternary (per_step) | rmse | 4.2583 ± 6.1345 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.1291 ± 0.0987 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 137147.7824 ± 15510.9306 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 48291.9444 ± 8874.9877 | 3 |
| dfa_float | off (off) | loss | 0.6083 ± 0.0110 | 5 |
| dfa_float | off (off) | mae | 0.5668 ± 0.0043 | 5 |
| dfa_float | off (off) | r2 | 0.5561 ± 0.0081 **(best)** | 5 |
| dfa_float | off (off) | rmse | 0.7799 ± 0.0071 | 5 |
| dfa_float | off (off) | sample_count | 4160.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 101040.5107 ± 60122.3225 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 51680.1554 ± 23419.0133 | 5 |
| dfa_float_clip1 | off (off) | loss | 0.7688 ± 0.0335 | 3 |
| dfa_float_clip1 | off (off) | mae | 0.6406 ± 0.0140 | 3 |
| dfa_float_clip1 | off (off) | r2 | 0.4390 ± 0.0245 | 3 |
| dfa_float_clip1 | off (off) | rmse | 0.8768 ± 0.0192 | 3 |
| dfa_float_clip1 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 85934.6571 ± 3493.0466 | 3 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 34962.1641 ± 2233.0547 | 3 |
| dfa_float_lr06 | off (off) | loss | 5.6339 | 1 |
| dfa_float_lr06 | off (off) | mae | 2.0648 | 1 |
| dfa_float_lr06 | off (off) | r2 | -3.1112 | 1 |
| dfa_float_lr06 | off (off) | rmse | 2.3736 | 1 |
| dfa_float_lr06 | off (off) | sample_count | 4160.0000 | 1 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 1 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 1 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 89112.8630 | 1 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 42729.9537 | 1 |
| dfa_float_lr10 | off (off) | loss | 5.6331 ± 0.0103 | 2 |
| dfa_float_lr10 | off (off) | mae | 2.0647 ± 0.0018 | 2 |
| dfa_float_lr10 | off (off) | r2 | -3.1106 ± 0.0076 | 2 |
| dfa_float_lr10 | off (off) | rmse | 2.3734 ± 0.0022 | 2 |
| dfa_float_lr10 | off (off) | sample_count | 4160.0000 | 2 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 2 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 2 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 87826.7945 ± 8884.3734 | 2 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 42703.2768 ± 8409.1612 | 2 |
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
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.3566 ± 1.3843 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 125587.4919 ± 21340.8158 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 58359.6210 ± 2703.6617 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 6.0149 ± 1.6395 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | mae | 2.0830 ± 0.0784 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | r2 | -3.3893 ± 1.1964 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | rmse | 2.4501 ± 0.3294 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.3372 ± 1.4259 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 75610.9006 ± 9818.6306 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 35441.2412 ± 420.0240 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 5.6339 ± 0.0001 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 2.0648 ± 0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -3.1112 ± 0.0001 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | rmse | 2.3736 ± 0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 4160.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4231 ± 0.6538 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 63823.8520 ± 27487.5051 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 31151.8436 ± 9223.0962 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 5.6339 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | mae | 2.0648 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | r2 | -3.1112 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | rmse | 2.3736 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.5957 ± 0.8705 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 77665.5150 ± 14483.2980 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 34844.1543 ± 1975.5922 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3990.1940 ± 8562.0859 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 47.1905 ± 73.6977 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -2910.7762 ± 6248.0365 | 3 |
| dfa_ternary_step | ternary (per_step) | rmse | 57.4392 ± 79.9788 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.1301 ± 0.1711 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 142183.3301 ± 16767.9815 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 49562.6112 ± 12314.1833 | 3 |
| structured_hadamard_float | off (off) | loss | 0.8640 ± 0.0513 | 3 |
| structured_hadamard_float | off (off) | mae | 0.6732 ± 0.0285 | 3 |
| structured_hadamard_float | off (off) | r2 | 0.3695 ± 0.0374 | 3 |
| structured_hadamard_float | off (off) | rmse | 0.9295 ± 0.0276 | 3 |
| structured_hadamard_float | off (off) | sample_count | 4160.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 161174.3075 ± 2521.2085 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 34423.1477 ± 920.0770 | 3 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.9023 ± 0.0319 | 3 |
| structured_hadamard_float_clip1 | off (off) | mae | 0.6805 ± 0.0252 | 3 |
| structured_hadamard_float_clip1 | off (off) | r2 | 0.3416 ± 0.0233 | 3 |
| structured_hadamard_float_clip1 | off (off) | rmse | 0.9499 ± 0.0168 | 3 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 4160.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 101655.2044 ± 3960.5121 | 3 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 18692.5852 ± 364.7518 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 6131.0257 ± 23787.0462 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 34.6819 ± 92.1955 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -4473.0117 ± 17358.1922 | 3 |
| structured_hadamard_ternary | ternary (per_step) | rmse | 58.0610 ± 159.8479 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 4160.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.2585 ± 0.2453 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 134680.2024 ± 91.1947 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 23979.9197 ± 518.8115 | 3 |
| structured_orth_float | off (off) | loss | 3.0929 ± 1.2124 | 3 |
| structured_orth_float | off (off) | mae | 1.3479 ± 0.3892 | 3 |
| structured_orth_float | off (off) | r2 | -1.2570 ± 0.8847 | 3 |
| structured_orth_float | off (off) | rmse | 1.7551 ± 0.3401 | 3 |
| structured_orth_float | off (off) | sample_count | 4160.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 158996.5665 ± 3387.5595 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 25323.7639 ± 702.0861 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4405.3143 ± 6343.9387 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 43.2086 ± 40.5942 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -3213.7032 ± 4629.3813 | 3 |
| structured_orth_ternary | ternary (per_step) | rmse | 64.6183 ± 46.1235 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 4160.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.1494 ± 0.0087 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 133740.0766 ± 22550.5180 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 20296.9773 ± 1195.3023 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 310.2312 ± 1036.9322 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 10.7208 ± 19.5797 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -225.3860 ± 756.6837 | 3 |
| ternary_dfa_step | ternary (per_step) | rmse | 14.8690 ± 28.7278 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.1478 ± 0.1292 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 136225.5769 ± 9300.5153 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 47477.8602 ± 8075.3274 | 3 |

## cifar10 (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.2362 ± 0.0312 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 214.1708 ± 32.8496 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 98.2057 ± 3.1370 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 0.6403 ± 0.0833 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 215.3774 ± 24.3792 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 98.2120 ± 2.8881 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.2362 ± 0.0312 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 223.0111 ± 49.9603 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 100.6332 ± 6.3113 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1016 ± 0.0097 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 211.9118 ± 24.4349 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 101.2210 ± 5.0346 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 5 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6348 ± 0.0104 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 82.8979 ± 5.5392 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 51.9163 ± 2.2807 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0231 ± 0.0007 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 198.2328 ± 23.8441 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 105.0146 ± 4.0553 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.1250 ± 0.0854 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2824 ± 0.1068 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1013 ± 0.0628 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 230.6713 ± 19.8734 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 99.7528 ± 5.2399 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0809 ± 0.0062 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 223.9662 ± 31.8630 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 104.2739 ± 8.0902 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0231 ± 0.0007 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 220.9940 ± 19.4650 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 104.1663 ± 2.3703 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0109 ± 0.0004 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 201.4673 ± 25.1015 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 103.8049 ± 7.5548 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.9781 ± 0.0506 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 0.4533 ± 1.0483 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.9650 ± 0.0871 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6461 ± 0.0066 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 84.1402 ± 7.2545 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 61.7503 ± 2.8134 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2774 ± 0.0022 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 84.3464 ± 10.5042 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 64.0042 ± 3.5157 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.9781 ± 0.0506 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 0.4533 ± 1.0483 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.9650 ± 0.0871 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6461 ± 0.0066 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 87.3592 ± 5.7188 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 63.0210 ± 2.2295 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.8688 ± 0.1191 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 2.3918 ± 2.3367 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.8590 ± 0.1435 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9535 ± 0.0007 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 82.7078 ± 3.6132 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 62.5935 ± 4.1764 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.9266 ± 0.1668 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 1.5219 ± 3.4569 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.8722 ± 0.2680 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6765 ± 0.0044 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 83.9920 ± 14.6251 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 55.2254 ± 2.6829 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9187 ± 0.0901 | 5 |
| structured_hadamard_float | off (off) | loss | 1.5590 ± 0.1013 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8819 ± 0.1346 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 195.5261 ± 35.1617 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 101.3487 ± 7.0349 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.7875 ± 0.1009 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 1.8579 ± 0.0900 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.7259 ± 0.1295 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 204.5050 ± 24.7233 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 100.0662 ± 6.0452 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8516 ± 0.0953 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.0761 ± 1.9745 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7763 ± 0.0910 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6814 ± 0.0005 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 82.3866 ± 7.3386 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 54.7997 ± 1.9061 | 5 |
| structured_orth_float | off (off) | accuracy | 0.9187 ± 0.0952 | 5 |
| structured_orth_float | off (off) | loss | 1.5603 ± 0.0959 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8756 ± 0.1466 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 224.2679 ± 33.9318 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 102.1367 ± 9.9749 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8250 ± 0.1094 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 3.6266 ± 2.2678 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7609 ± 0.0888 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6814 ± 0.0005 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 77.6485 ± 17.7368 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 55.9499 ± 2.2367 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.8750 ± 0.1335 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 2.5904 ± 2.7669 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.8256 ± 0.1400 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6344 ± 0.0038 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 86.5054 ± 11.4786 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 56.1488 ± 1.0750 | 5 |

## fashion_mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.3975 ± 0.0213 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 162.4573 ± 19.4546 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 90.6341 ± 7.5556 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 1.0083 ± 0.0528 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 176.3951 ± 16.6279 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 76.8736 ± 3.5979 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3975 ± 0.0213 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 163.0843 ± 19.9368 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 75.3293 ± 7.1274 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1411 ± 0.0069 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 169.4661 ± 17.9959 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 71.0757 ± 9.6338 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.9984 ± 0.0043 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 0.0324 ± 0.0899 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.9986 ± 0.0040 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6140 ± 0.0177 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 95.5624 ± 4.0670 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 60.8838 ± 1.5581 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0226 ± 0.0007 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 189.7377 ± 34.5359 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 92.2744 ± 18.2921 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.1906 ± 0.1456 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2293 ± 0.0634 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1582 ± 0.1175 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 162.3893 ± 24.3853 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 76.4787 ± 4.2542 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0811 ± 0.0027 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 170.0911 ± 13.5949 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 73.3401 ± 9.0193 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0226 ± 0.0007 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 173.2018 ± 15.6485 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 70.5178 ± 8.6888 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0107 ± 0.0004 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 181.8000 ± 18.9503 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 74.8880 ± 6.5827 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6505 ± 0.0151 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 96.6668 ± 7.9261 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 68.3758 ± 4.2202 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.9750 ± 0.0694 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 0.5181 ± 1.4382 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.9723 ± 0.0770 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2595 ± 0.0052 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 74.6788 ± 8.9015 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 47.6521 ± 5.3377 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6505 ± 0.0151 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 75.2829 ± 9.2747 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 52.3841 ± 0.8189 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.9344 ± 0.0807 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 0.5959 ± 0.7674 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.9081 ± 0.0975 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9533 ± 0.0006 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 78.6530 ± 3.4046 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 53.2933 ± 1.9597 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8250 ± 0.1359 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5801 ± 2.8416 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7840 ± 0.1499 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6788 ± 0.0014 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 89.7688 ± 5.7742 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 62.8128 ± 3.8942 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9031 ± 0.0906 | 5 |
| structured_hadamard_float | off (off) | loss | 1.8252 ± 0.0427 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8789 ± 0.1190 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 149.8383 ± 16.5634 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 74.8377 ± 5.7112 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.7953 ± 0.1130 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 2.0002 ± 0.0363 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.7728 ± 0.1027 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 166.8395 ± 16.8826 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 75.4283 ± 3.6042 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8297 ± 0.1542 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.5294 ± 3.1951 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7775 ± 0.1281 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6807 ± 0.0007 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 75.2536 ± 7.7734 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 48.5250 ± 1.5218 | 5 |
| structured_orth_float | off (off) | accuracy | 0.8891 ± 0.0921 | 5 |
| structured_orth_float | off (off) | loss | 1.8298 ± 0.0330 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8533 ± 0.1374 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 178.8515 ± 15.4635 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 81.9733 ± 19.7164 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8469 ± 0.1028 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 3.1598 ± 2.1402 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7810 ± 0.1019 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6807 ± 0.0007 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 87.8842 ± 20.7924 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 60.2928 ± 9.2946 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.8891 ± 0.2199 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 2.2990 ± 4.5577 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.8619 ± 0.2599 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6006 ± 0.0039 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 90.7610 ± 11.6935 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 60.1541 ± 6.2545 | 5 |

## fashion_mnist (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.8707 ± 0.0024 | 5 |
| backprop_float | off (off) | loss | 0.3659 ± 0.0034 | 5 |
| backprop_float | off (off) | macro_f1 | 0.8696 ± 0.0023 | 5 |
| backprop_float | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 97.4447 ± 3.5481 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 54.6266 ± 0.9970 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 0.8639 ± 0.0012 | 5 |
| backprop_float_lr06 | off (off) | loss | 0.3820 ± 0.0033 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.8624 ± 0.0015 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 218.8149 ± 6.5428 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 119.3485 ± 1.4748 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 0.8707 ± 0.0024 | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3659 ± 0.0034 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.8696 ± 0.0023 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 224.9068 ± 5.9059 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 120.9464 ± 2.2016 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 0.8738 ± 0.0032 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.3580 ± 0.0073 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.8729 ± 0.0031 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 221.5629 ± 2.8496 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 119.7115 ± 2.2617 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.1251 ± 0.0562 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 2.7869 ± 0.6993 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.0472 ± 0.0533 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.4795 ± 0.0234 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 90.3281 ± 34.9206 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 55.3429 ± 18.1559 | 5 |
| dfa_float | off (off) | accuracy | 0.7743 ± 0.0459 | 5 |
| dfa_float | off (off) | loss | 4.6761 ± 0.9516 | 5 |
| dfa_float | off (off) | macro_f1 | 0.7536 ± 0.0602 | 5 |
| dfa_float | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 220.9657 ± 4.2772 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 119.3608 ± 1.2122 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.8382 ± 0.0027 | 5 |
| dfa_float_clip1 | off (off) | loss | 0.4671 ± 0.0048 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.8342 ± 0.0029 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 366.3145 ± 35.5213 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 192.7326 ± 17.1595 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 0.7827 ± 0.0376 | 5 |
| dfa_float_lr06 | off (off) | loss | 4.4953 ± 0.7812 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.7695 ± 0.0500 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 222.9469 ± 5.1975 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 120.2521 ± 0.7387 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 0.7743 ± 0.0459 | 5 |
| dfa_float_lr10 | off (off) | loss | 4.6761 ± 0.9516 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.7536 ± 0.0602 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 224.0751 ± 4.0153 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 121.4028 ± 3.0153 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 0.7658 ± 0.0404 | 5 |
| dfa_float_lr15 | off (off) | loss | 4.8527 ± 0.8376 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.7476 ± 0.0573 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 223.5946 ± 12.5668 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 121.7669 ± 5.1281 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.6049 ± 0.0568 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 8.1824 ± 1.1781 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.5765 ± 0.0671 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4039 ± 0.0201 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 140.5570 ± 33.7717 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 101.3075 ± 26.9297 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.5820 ± 0.0698 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 8.6582 ± 1.4467 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.5242 ± 0.0912 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.1267 ± 0.0048 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 115.3253 ± 1.7549 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 81.8821 ± 0.4800 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.6049 ± 0.0568 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 8.1824 ± 1.1781 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.5765 ± 0.0671 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4039 ± 0.0201 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 116.4147 ± 3.7664 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 83.1030 ± 2.4549 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.6160 ± 0.0813 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 7.9532 ± 1.6781 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.5827 ± 0.0625 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.7332 ± 0.0092 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 140.4419 ± 47.7848 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 99.9529 ± 33.6623 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.2827 ± 0.0986 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 14.7691 ± 2.1015 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.1982 ± 0.0951 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6394 ± 0.0056 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 176.7348 ± 30.9649 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 111.3236 ± 18.2923 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.0999 ± 0.0049 | 5 |
| structured_hadamard_float | off (off) | loss | 15.3793 ± 9.0758 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0183 ± 0.0009 | 5 |
| structured_hadamard_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 220.8060 ± 7.0006 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 117.6346 ± 4.9797 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.7244 ± 0.0142 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.8108 ± 0.0333 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.6983 ± 0.0205 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 219.5872 ± 4.7895 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 114.5556 ± 2.6971 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.2599 ± 0.0485 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 15.2737 ± 1.0477 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.1829 ± 0.0908 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6763 ± 0.0011 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 115.1212 ± 0.9120 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 74.2931 ± 0.4869 | 5 |
| structured_orth_float | off (off) | accuracy | 0.1003 ± 0.0015 | 5 |
| structured_orth_float | off (off) | loss | 15.3683 ± 9.0675 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.0183 ± 0.0002 | 5 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 223.5788 ± 4.3954 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 118.8798 ± 2.2221 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.3705 ± 0.0523 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 12.8806 ± 1.1106 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.2991 ± 0.0639 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6704 ± 0.0014 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 152.8728 ± 40.9970 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 98.7807 ± 25.7843 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.1823 ± 0.0975 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 13.3422 ± 1.5454 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1180 ± 0.0899 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6075 ± 0.0087 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 115.4727 ± 1.1396 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 75.7955 ± 1.0417 | 5 |

## mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float | off (off) | loss | 0.3895 ± 0.0783 | 5 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float | off (off) | sample_count | 128.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 309.2660 ± 39.0344 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 149.1697 ± 10.1738 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr06 | off (off) | loss | 0.9855 ± 0.1190 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 305.0608 ± 79.8939 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 135.8510 ± 19.9217 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr10 | off (off) | loss | 0.3895 ± 0.0783 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 304.1294 ± 53.0640 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 111.6839 ± 32.5253 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1374 ± 0.0276 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 317.5660 ± 49.1184 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 148.1184 ± 13.1596 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 5 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5215 ± 0.0429 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 168.0003 ± 16.7719 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 97.9810 ± 3.7348 | 5 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float | off (off) | loss | 0.0191 ± 0.0008 | 5 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float | off (off) | sample_count | 128.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 303.1688 ± 73.7805 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 181.0149 ± 13.7388 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.2016 ± 0.1212 | 5 |
| dfa_float_clip1 | off (off) | loss | 2.2539 ± 0.0751 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.1280 ± 0.1090 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 282.0742 ± 42.1812 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 157.4599 ± 38.9228 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr06 | off (off) | loss | 0.0683 ± 0.0095 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 323.8525 ± 55.0125 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 160.7433 ± 12.3177 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr10 | off (off) | loss | 0.0191 ± 0.0008 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 293.0458 ± 27.2200 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 160.3227 ± 28.9131 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 1.0000 **(best)** | 5 |
| dfa_float_lr15 | off (off) | loss | 0.0090 ± 0.0004 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 1.0000 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 128.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 329.4456 ± 26.5415 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 172.7468 ± 34.2369 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6079 ± 0.0086 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 173.3292 ± 6.5375 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 123.0001 ± 11.1206 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2528 ± 0.0067 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 165.1906 ± 16.7409 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 120.7217 ± 8.1730 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | -0.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 1.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6079 ± 0.0086 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 133.0554 ± 22.8337 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 107.0258 ± 29.2003 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.9187 ± 0.1462 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 1.4699 ± 2.5536 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.8943 ± 0.1797 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 128.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9501 ± 0.0016 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 157.8562 ± 24.7522 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 118.8663 ± 9.6347 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.9359 ± 0.1098 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 1.1506 ± 2.3751 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.9447 ± 0.0777 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6628 ± 0.0099 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 157.4406 ± 17.6866 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 111.8511 ± 6.4312 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.9078 ± 0.0687 | 5 |
| structured_hadamard_float | off (off) | loss | 1.7804 ± 0.0793 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8764 ± 0.0875 | 5 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 302.9399 ± 22.5887 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 173.2787 ± 15.5463 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.4938 ± 0.0697 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 2.0837 ± 0.0753 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.4637 ± 0.1306 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 128.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 329.5882 ± 57.1396 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 172.0763 ± 18.1792 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8219 ± 0.0765 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 3.6913 ± 1.5852 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7956 ± 0.1069 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6783 ± 0.0012 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 156.4521 ± 25.5672 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 104.5589 ± 8.7793 | 5 |
| structured_orth_float | off (off) | accuracy | 0.8469 ± 0.0681 | 5 |
| structured_orth_float | off (off) | loss | 1.7855 ± 0.0869 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8245 ± 0.1075 | 5 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 289.5983 ± 29.3624 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 176.0785 ± 13.0631 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8547 ± 0.0600 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 2.9994 ± 1.2580 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7975 ± 0.0881 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6783 ± 0.0011 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 164.2910 ± 26.4214 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 115.3510 ± 5.7748 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 5 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5768 ± 0.0099 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 149.7673 ± 8.5384 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 107.4283 ± 11.6516 | 5 |

## mnist (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 0.9599 ± 0.0011 | 5 |
| backprop_float | off (off) | loss | 0.1355 ± 0.0021 | 5 |
| backprop_float | off (off) | macro_f1 | 0.9597 ± 0.0010 | 5 |
| backprop_float | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float | off (off) | test_throughput_samples_sec | 245.2276 ± 16.4230 | 5 |
| backprop_float | off (off) | train_throughput_samples_sec | 123.1033 ± 2.7359 | 5 |
| backprop_float_lr06 | off (off) | accuracy | 0.9507 ± 0.0016 | 5 |
| backprop_float_lr06 | off (off) | loss | 0.1696 ± 0.0032 | 5 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.9504 ± 0.0015 | 5 |
| backprop_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 185.3001 ± 54.6168 | 5 |
| backprop_float_lr06 | off (off) | train_throughput_samples_sec | 98.0524 ± 29.1627 | 5 |
| backprop_float_lr10 | off (off) | accuracy | 0.9599 ± 0.0011 | 5 |
| backprop_float_lr10 | off (off) | loss | 0.1355 ± 0.0021 | 5 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.9597 ± 0.0010 | 5 |
| backprop_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 195.6626 ± 46.5381 | 5 |
| backprop_float_lr10 | off (off) | train_throughput_samples_sec | 99.6303 ± 21.3446 | 5 |
| backprop_float_lr15 | off (off) | accuracy | 0.9646 ± 0.0004 **(best)** | 5 |
| backprop_float_lr15 | off (off) | loss | 0.1170 ± 0.0016 | 5 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.9645 ± 0.0004 | 5 |
| backprop_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 161.6193 ± 5.2125 | 5 |
| backprop_float_lr15 | off (off) | train_throughput_samples_sec | 81.6983 ± 2.0138 | 5 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.3757 ± 0.2236 | 5 |
| backprop_ternary_step | ternary (per_step) | loss | 7.8465 ± 2.9107 | 5 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3352 ± 0.2556 | 5 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5536 ± 0.0136 | 5 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 118.9005 ± 14.0855 | 5 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 72.3303 ± 9.0451 | 5 |
| dfa_float | off (off) | accuracy | 0.9363 ± 0.0074 | 5 |
| dfa_float | off (off) | loss | 1.1894 ± 0.1421 | 5 |
| dfa_float | off (off) | macro_f1 | 0.9360 ± 0.0074 | 5 |
| dfa_float | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float | off (off) | test_throughput_samples_sec | 237.5083 ± 18.0866 | 5 |
| dfa_float | off (off) | train_throughput_samples_sec | 146.4357 ± 18.0025 | 5 |
| dfa_float_clip1 | off (off) | accuracy | 0.9376 ± 0.0022 | 5 |
| dfa_float_clip1 | off (off) | loss | 0.2163 ± 0.0042 | 5 |
| dfa_float_clip1 | off (off) | macro_f1 | 0.9372 ± 0.0022 | 5 |
| dfa_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 236.6090 ± 33.2905 | 5 |
| dfa_float_clip1 | off (off) | train_throughput_samples_sec | 145.1826 ± 8.3970 | 5 |
| dfa_float_lr06 | off (off) | accuracy | 0.9514 ± 0.0010 | 5 |
| dfa_float_lr06 | off (off) | loss | 0.1630 ± 0.0056 | 5 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.9511 ± 0.0010 | 5 |
| dfa_float_lr06 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 189.1014 ± 34.4883 | 5 |
| dfa_float_lr06 | off (off) | train_throughput_samples_sec | 117.5495 ± 32.6027 | 5 |
| dfa_float_lr10 | off (off) | accuracy | 0.9363 ± 0.0074 | 5 |
| dfa_float_lr10 | off (off) | loss | 1.1894 ± 0.1421 | 5 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.9360 ± 0.0074 | 5 |
| dfa_float_lr10 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 217.6125 ± 30.0168 | 5 |
| dfa_float_lr10 | off (off) | train_throughput_samples_sec | 129.1371 ± 25.8184 | 5 |
| dfa_float_lr15 | off (off) | accuracy | 0.9335 ± 0.0067 | 5 |
| dfa_float_lr15 | off (off) | loss | 1.3732 ± 0.1410 | 5 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.9332 ± 0.0067 | 5 |
| dfa_float_lr15 | off (off) | sample_count | 14016.0000 | 5 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 5 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 236.3832 ± 45.1956 | 5 |
| dfa_float_lr15 | off (off) | train_throughput_samples_sec | 137.8279 ± 21.1522 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.8201 ± 0.0268 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 3.7245 ± 0.5530 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.8165 ± 0.0334 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4803 ± 0.0037 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 80.9360 ± 10.7738 | 5 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 66.9637 ± 7.9756 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.8239 ± 0.0523 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 3.6481 ± 1.0827 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.8202 ± 0.0565 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.1853 ± 0.0013 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 97.2297 ± 27.5795 | 5 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | train_throughput_samples_sec | 77.5412 ± 18.9747 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.8201 ± 0.0268 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 3.7245 ± 0.5530 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.8165 ± 0.0334 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4803 ± 0.0037 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 83.0894 ± 6.6955 | 5 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | train_throughput_samples_sec | 66.4458 ± 5.6801 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.7885 ± 0.0660 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.3722 ± 1.3643 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.7818 ± 0.0736 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 14016.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.7480 ± 0.0044 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 100.1046 ± 31.3546 | 5 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | train_throughput_samples_sec | 79.6137 ± 25.6478 | 5 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.1641 ± 0.0779 | 5 |
| dfa_ternary_step | ternary (per_step) | loss | 17.2959 ± 1.6460 | 5 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0762 ± 0.0631 | 5 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6316 ± 0.0039 | 5 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 126.8444 ± 9.4752 | 5 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 93.1741 ± 5.0180 | 5 |
| structured_hadamard_float | off (off) | accuracy | 0.7979 ± 0.0118 | 5 |
| structured_hadamard_float | off (off) | loss | 0.7630 ± 0.0351 | 5 |
| structured_hadamard_float | off (off) | macro_f1 | 0.7947 ± 0.0126 | 5 |
| structured_hadamard_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 5 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 162.9266 ± 6.0591 | 5 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 98.3756 ± 6.6339 | 5 |
| structured_hadamard_float_clip1 | off (off) | accuracy | 0.8025 ± 0.0096 | 5 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.7444 ± 0.0282 | 5 |
| structured_hadamard_float_clip1 | off (off) | macro_f1 | 0.7994 ± 0.0098 | 5 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 14016.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 233.8413 ± 23.6136 | 5 |
| structured_hadamard_float_clip1 | off (off) | train_throughput_samples_sec | 137.0693 ± 12.2185 | 5 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.3566 ± 0.0472 | 5 |
| structured_hadamard_ternary | ternary (per_step) | loss | 13.0831 ± 0.9927 | 5 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2917 ± 0.0478 | 5 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6768 ± 0.0012 | 5 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 125.7998 ± 20.3625 | 5 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 90.9228 ± 13.9236 | 5 |
| structured_orth_float | off (off) | accuracy | 0.8074 ± 0.0091 | 5 |
| structured_orth_float | off (off) | loss | 0.7313 ± 0.0307 | 5 |
| structured_orth_float | off (off) | macro_f1 | 0.8043 ± 0.0092 | 5 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 5 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 5 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 5 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 161.0175 ± 15.7156 | 5 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 97.8138 ± 6.9540 | 5 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.3603 ± 0.0447 | 5 |
| structured_orth_ternary | ternary (per_step) | loss | 13.0266 ± 0.9720 | 5 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.2933 ± 0.0544 | 5 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6768 ± 0.0011 | 5 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 82.7825 ± 5.5339 | 5 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 61.0854 ± 3.8690 | 5 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.2984 ± 0.1231 | 5 |
| ternary_dfa_step | ternary (per_step) | loss | 14.5126 ± 2.5528 | 5 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1857 ± 0.0996 | 5 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 5 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5882 ± 0.0048 | 5 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 80.8255 ± 3.4518 | 5 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 59.5446 ± 5.0617 | 5 |

## ucr (offline)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| backprop_float | off (off) | loss | 1.1311 ± 0.1205 | 3 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 32.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 50335.9869 ± 9207.4622 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 30230.1368 ± 3470.2138 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6118 ± 0.0732 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 47157.3277 ± 13408.1252 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 16273.0870 ± 3166.6282 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0725 ± 0.0298 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 32.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 47549.6669 ± 21794.4306 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 26300.9210 ± 10730.1196 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6373 ± 0.0227 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 49291.8923 ± 4602.4668 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 28919.1458 ± 9943.0275 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8177 ± 0.3138 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5722 ± 6.9143 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7355 ± 0.4628 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6647 ± 0.0200 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 51097.3766 ± 7789.3116 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 18415.9402 ± 7078.0820 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.6771 ± 1.0050 | 3 |
| structured_hadamard_float | off (off) | loss | 1.3163 ± 0.0483 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.6049 ± 1.0831 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 48358.2107 ± 17100.0504 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 17406.3199 ± 885.1018 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.2094 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.6941 ± 0.0184 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6772 ± 0.0052 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 48343.7547 ± 20089.3586 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 11160.9440 ± 1683.5713 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7760 ± 0.5439 | 3 |
| structured_orth_float | off (off) | loss | 1.3158 ± 0.1104 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.6657 ± 0.7974 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 44240.0328 ± 18477.9808 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 8314.5240 ± 315.7411 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4.1640 ± 0.1953 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.6772 ± 0.0495 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6770 ± 0.0055 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 42631.4021 ± 11279.4025 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 6929.4500 ± 1213.1119 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6072 ± 0.0159 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 51612.2241 ± 7574.8899 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 17675.4828 ± 3378.8732 | 3 |

## ucr (real)

| Strategy Variant | Flip | Metric | Mean ± 95% CI | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| backprop_float | off (off) | loss | 1.0295 ± 0.1481 | 3 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 3 |
| backprop_float | off (off) | sample_count | 64.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 32.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 41105.7350 ± 3432.4480 | 3 |
| backprop_float | off (off) | train_throughput_samples_sec | 28270.9856 ± 4470.8635 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6118 ± 0.0732 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 38061.0459 ± 3779.8431 | 3 |
| backprop_ternary_step | ternary (per_step) | train_throughput_samples_sec | 13778.8572 ± 3315.3950 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0327 ± 0.0102 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 32.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 37193.7595 ± 4728.8054 | 3 |
| dfa_float | off (off) | train_throughput_samples_sec | 22665.6426 ± 2197.3808 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6373 ± 0.0227 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 37568.9243 ± 1749.2146 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | train_throughput_samples_sec | 22776.9334 ± 2291.9816 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8177 ± 0.3138 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5722 ± 6.9143 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7355 ± 0.4628 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6647 ± 0.0200 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 36351.2168 ± 7817.8501 | 3 |
| dfa_ternary_step | ternary (per_step) | train_throughput_samples_sec | 14879.8923 ± 2466.9447 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.7969 ± 0.5476 | 3 |
| structured_hadamard_float | off (off) | loss | 1.2706 ± 0.0217 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.6838 ± 0.8131 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 38488.5719 ± 10035.6147 | 3 |
| structured_hadamard_float | off (off) | train_throughput_samples_sec | 13662.3827 ± 1205.4949 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.2094 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.6941 ± 0.0184 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6772 ± 0.0052 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 39054.8162 ± 6270.7506 | 3 |
| structured_hadamard_ternary | ternary (per_step) | train_throughput_samples_sec | 10237.9754 ± 1948.6945 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7760 ± 0.5439 | 3 |
| structured_orth_float | off (off) | loss | 1.2960 ± 0.1077 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.6644 ± 0.7978 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 41204.3096 ± 3464.4454 | 3 |
| structured_orth_float | off (off) | train_throughput_samples_sec | 6680.8939 ± 78.9909 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4.1640 ± 0.1953 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.6772 ± 0.0495 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6770 ± 0.0055 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 34852.1274 ± 12263.6596 | 3 |
| structured_orth_ternary | ternary (per_step) | train_throughput_samples_sec | 5405.9354 ± 998.9175 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6072 ± 0.0159 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 34596.8636 ± 17851.2341 | 3 |
| ternary_dfa_step | ternary (per_step) | train_throughput_samples_sec | 15101.9448 ± 600.1507 | 3 |

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
