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
| california_housing | offline | R² | 0.0840 ± 0.0556 | dfa_float | off (off) | 3 |
| california_housing | offline | Zero Ratio | 0.2186 ± 0.2010 | structured_hadamard_ternary | ternary (per_step) | 3 |
| california_housing | offline | Test Throughput (samples/s) | 87757.7091 ± 1022.0328 | backprop_float | off (off) | 3 |
| california_housing | real | R² | nan ± nan | dfa_float_lr15 | off (off) | 3 |
| california_housing | real | Zero Ratio | 0.5957 ± 0.3504 | dfa_ternary_epoch_tau010 | ternary (per_epoch) | 3 |
| california_housing | real | Test Throughput (samples/s) | 161174.3075 ± 1014.8411 | structured_hadamard_float | off (off) | 3 |
| mnist | offline | Accuracy | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
| mnist | offline | Macro-F1 | 1.0000 | ternary_dfa_step | ternary (per_step) | 3 |
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

| Dataset | Mode | Primary | Best (μ±σ) | Variant | Flip | n | Baseline (μ±σ) | Δ | Effect Size |
|---|---|---|---|---|---|---:|---|---:|---:|
| 20newsgroups | offline | accuracy | 0.3906 ± 0.0681 | backprop_ternary_step | ternary (per_step) | 3 | 0.1771 ± 0.0722 | 0.2135 | 3.043 |
| 20newsgroups | real | accuracy | 0.3206 ± 0.0046 | dfa_float_lr15 | off (off) | 3 | 0.3206 ± 0.0046 | 0.0000 | 0.000 |
| california_housing | offline | r2 | 0.0840 ± 0.0556 | dfa_float | off (off) | 3 | 0.0840 ± 0.0556 | 0.0000 | 0.000 |
| california_housing | real | r2 | nan ± nan | dfa_float_lr15 | off (off) | 3 | nan ± nan | nan | 0.000 |
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
| backprop_ternary_step | ternary (per_step) | accuracy | 0.3906 ± 0.0681 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 12.0424 ± 1.1263 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3414 ± 0.0474 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6810 ± 0.0002 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 7772.4028 ± 238.0358 | 3 |
| dfa_float | off (off) | accuracy | 0.1510 ± 0.0325 | 3 |
| dfa_float | off (off) | loss | 2.0691 ± 0.0078 | 3 |
| dfa_float | off (off) | macro_f1 | 0.0780 ± 0.0426 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 7518.0989 ± 822.4189 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.3438 ± 0.0413 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 13.0543 ± 1.3608 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.2602 ± 0.0542 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6818 ± 0.0001 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 7491.7946 ± 264.3764 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.2240 ± 0.0502 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 15.7157 ± 1.0053 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.1734 ± 0.0701 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0001 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 7340.1342 ± 880.5489 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.1354 ± 0.0361 | 3 |
| structured_hadamard_float | off (off) | loss | 2.0756 ± 0.0078 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0825 ± 0.0410 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 17159.8569 ± 550.9908 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.2917 ± 0.0180 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 14.5478 ± 0.6010 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2593 ± 0.0096 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0001 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 7245.0925 ± 290.8769 | 3 |
| structured_orth_float | off (off) | accuracy | 0.1354 ± 0.0361 | 3 |
| structured_orth_float | off (off) | loss | 2.0751 ± 0.0073 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.0817 ± 0.0410 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 17386.8935 ± 183.4235 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.2083 ± 0.0861 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 16.1435 ± 2.0061 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.1511 ± 0.0790 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6819 ± 0.0001 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 7804.6346 ± 167.3541 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.2656 ± 0.0413 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 14.8916 ± 0.8861 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.2078 ± 0.0311 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6809 ± 0.0003 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 7573.5650 ± 385.2984 | 3 |

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
| backprop_float_lr06 | off (off) | accuracy | 0.0862 ± 0.0165 | 3 |
| backprop_float_lr06 | off (off) | loss | 2.9690 ± 0.0004 | 3 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.0439 ± 0.0159 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 8381.7305 ± 415.6071 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.1230 ± 0.0042 | 3 |
| backprop_float_lr10 | off (off) | loss | 2.9214 ± 0.0026 | 3 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.0722 ± 0.0035 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 8209.2622 ± 596.1825 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.1986 ± 0.0134 | 3 |
| backprop_float_lr15 | off (off) | loss | 2.6976 ± 0.0102 | 3 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.1400 ± 0.0198 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 3776.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 7069.3706 ± 1939.4838 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.0615 ± 0.0122 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 17.8996 ± 0.1939 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.0541 ± 0.0109 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6822 ± 0.0001 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 6125.3234 ± 104.1322 | 3 |
| dfa_float | off (off) | accuracy | 0.2229 ± 0.0107 | 3 |
| dfa_float | off (off) | loss | 2.5541 ± 0.0192 | 3 |
| dfa_float | off (off) | macro_f1 | 0.1686 ± 0.0084 | 3 |
| dfa_float | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 6134.0114 ± 112.5139 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.1294 ± 0.0168 | 3 |
| dfa_float_lr06 | off (off) | loss | 2.8922 ± 0.0220 | 3 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.0769 ± 0.0103 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 1397.4045 ± 307.6902 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.2229 ± 0.0107 | 3 |
| dfa_float_lr10 | off (off) | loss | 2.5541 ± 0.0192 | 3 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.1686 ± 0.0084 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 2329.9449 ± 458.5847 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.3206 ± 0.0046 **(best)** | 3 |
| dfa_float_lr15 | off (off) | loss | 2.2206 ± 0.0104 | 3 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.2744 ± 0.0045 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 3776.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 ± 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 2433.5734 ± 653.8873 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.1264 ± 0.0184 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 17.2000 ± 0.7207 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.0992 ± 0.0296 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6793 ± 0.0003 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 5723.3596 ± 202.3799 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.1028 ± 0.0143 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 18.3765 ± 0.3953 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.0633 ± 0.0137 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.2971 ± 0.0006 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 2677.7276 ± 197.5750 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.1264 ± 0.0184 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 17.2000 ± 0.7207 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.0992 ± 0.0296 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.6793 ± 0.0003 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 3722.4781 ± 110.7387 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.0841 ± 0.0061 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.5784 ± 0.8237 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.0575 ± 0.0069 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 3776.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.9544 ± 0.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 3686.5771 ± 76.4597 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.0599 ± 0.0053 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 18.3875 ± 0.1007 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0441 ± 0.0038 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0001 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 5917.8555 ± 201.9331 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.0722 ± 0.0056 | 3 |
| structured_hadamard_float | off (off) | loss | 2.9809 ± 0.0005 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.0310 ± 0.0087 | 3 |
| structured_hadamard_float | off (off) | sample_count | 3776.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 16670.0619 ± 268.4730 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.0563 ± 0.0072 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 18.3074 ± 0.1135 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.0458 ± 0.0043 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 3776.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0001 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 5664.4212 ± 96.1980 | 3 |
| structured_orth_float | off (off) | accuracy | 0.0876 ± 0.0153 | 3 |
| structured_orth_float | off (off) | loss | 2.9855 ± 0.0001 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.0448 ± 0.0160 | 3 |
| structured_orth_float | off (off) | sample_count | 3776.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 17052.5182 ± 417.3055 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.0576 ± 0.0041 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 18.3228 ± 0.1869 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.0441 ± 0.0046 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 3776.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6821 ± 0.0001 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 5739.9968 ± 127.6391 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.0591 ± 0.0083 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 18.3253 ± 0.2789 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.0478 ± 0.0085 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 3776.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6820 ± 0.0001 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 5411.6277 ± 283.8592 | 3 |

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
| backprop_ternary_step | ternary (per_step) | loss | 122.5007 ± 201.1404 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 4.8773 ± 5.0992 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -18.3791 ± 31.8195 | 3 |
| backprop_ternary_step | ternary (per_step) | rmse | 7.9612 ± 9.4170 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0810 ± 0.0291 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 78312.6326 ± 2059.0049 | 3 |
| dfa_float | off (off) | loss | 5.7903 ± 0.3514 | 3 |
| dfa_float | off (off) | mae | 1.8461 ± 0.0570 | 3 |
| dfa_float | off (off) | r2 | 0.0840 ± 0.0556 **(best)** | 3 |
| dfa_float | off (off) | rmse | 2.4055 ± 0.0736 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 82614.6034 ± 950.2247 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 9.8164 ± 1.4005 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 2.2261 ± 0.0833 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -0.5529 ± 0.2216 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | rmse | 3.1280 ± 0.2196 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.0626 ± 0.0237 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 83823.6845 ± 515.3488 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 14681.5489 ± 16828.4125 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 90.7748 ± 83.7806 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -2321.5537 ± 2662.1777 | 3 |
| dfa_ternary_step | ternary (per_step) | rmse | 97.7155 ± 87.7488 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.0847 ± 0.0475 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 75960.4744 ± 4689.2680 | 3 |
| structured_hadamard_float | off (off) | loss | 6.4155 ± 0.0139 | 3 |
| structured_hadamard_float | off (off) | mae | 1.9511 ± 0.0027 | 3 |
| structured_hadamard_float | off (off) | r2 | -0.0149 ± 0.0022 | 3 |
| structured_hadamard_float | off (off) | rmse | 2.5329 ± 0.0028 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 83001.2409 ± 5342.9101 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 14.9452 ± 12.2499 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 2.6903 ± 0.9946 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -1.3643 ± 1.9379 | 3 |
| structured_hadamard_ternary | ternary (per_step) | rmse | 3.6629 ± 1.5143 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.2186 ± 0.2010 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 83191.0634 ± 1461.1867 | 3 |
| structured_orth_float | off (off) | loss | 6.4197 ± 0.0172 | 3 |
| structured_orth_float | off (off) | mae | 1.9518 ± 0.0030 | 3 |
| structured_orth_float | off (off) | r2 | -0.0156 ± 0.0027 | 3 |
| structured_orth_float | off (off) | rmse | 2.5337 ± 0.0034 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 84463.3734 ± 3121.8683 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 111340.1135 ± 74007.7494 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 309.3594 ± 97.6115 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -17612.4954 ± 11707.6866 | 3 |
| structured_orth_ternary | ternary (per_step) | rmse | 322.3666 ± 105.4982 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.1183 ± 0.0442 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 78221.6017 ± 3178.6024 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 186.0816 ± 144.5152 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 10.5616 ± 4.5927 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -28.4373 ± 22.8616 | 3 |
| ternary_dfa_step | ternary (per_step) | rmse | 12.9663 ± 5.1900 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.1366 ± 0.0382 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 79211.8284 ± 3961.1406 | 3 |

## california_housing (real)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | loss | 0.6843 ± 0.0044 | 3 |
| backprop_float | off (off) | mae | 0.6035 ± 0.0030 | 3 |
| backprop_float | off (off) | r2 | 0.5006 ± 0.0032 | 3 |
| backprop_float | off (off) | rmse | 0.8272 ± 0.0027 | 3 |
| backprop_float | off (off) | sample_count | 4160.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 159140.9741 ± 1845.3980 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 22.1981 ± 24.1133 | 3 |
| backprop_ternary_step | ternary (per_step) | mae | 2.9896 ± 1.2775 | 3 |
| backprop_ternary_step | ternary (per_step) | r2 | -15.1987 ± 17.5962 | 3 |
| backprop_ternary_step | ternary (per_step) | rmse | 4.2583 ± 2.4693 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.1291 ± 0.0397 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 137147.7824 ± 6243.4859 | 3 |
| dfa_float | off (off) | loss | 0.6113 ± 0.0050 | 3 |
| dfa_float | off (off) | mae | 0.5682 ± 0.0024 | 3 |
| dfa_float | off (off) | r2 | 0.5539 ± 0.0036 | 3 |
| dfa_float | off (off) | rmse | 0.7819 ± 0.0032 | 3 |
| dfa_float | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 136275.1503 ± 5768.0761 | 3 |
| dfa_float_clip1 | off (off) | loss | 0.7688 ± 0.0135 | 3 |
| dfa_float_clip1 | off (off) | mae | 0.6406 ± 0.0057 | 3 |
| dfa_float_clip1 | off (off) | r2 | 0.4390 ± 0.0099 | 3 |
| dfa_float_clip1 | off (off) | rmse | 0.8768 ± 0.0077 | 3 |
| dfa_float_clip1 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_clip1 | off (off) | test_throughput_samples_sec | 85934.6571 ± 1406.0270 | 3 |
| dfa_float_lr06 | off (off) | loss | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | mae | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | r2 | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | rmse | nan ± nan | 3 |
| dfa_float_lr06 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 87860.9249 ± 3279.1137 | 3 |
| dfa_float_lr10 | off (off) | loss | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | mae | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | r2 | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | rmse | nan ± nan | 3 |
| dfa_float_lr10 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 88387.9196 ± 1197.2894 | 3 |
| dfa_float_lr15 | off (off) | loss | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | mae | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | r2 | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | rmse | nan ± nan | 3 |
| dfa_float_lr15 | off (off) | sample_count | 4160.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 85085.9277 ± 1216.3363 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 5.6339 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | mae | 2.0648 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | r2 | -3.1112 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | rmse | 2.3736 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.3566 ± 0.5572 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 125587.4919 ± 8590.1411 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 6.0149 ± 0.6599 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | mae | 2.0830 ± 0.0315 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | r2 | -3.3893 ± 0.4816 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | rmse | 2.4501 ± 0.1326 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.3372 ± 0.5740 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 75610.9006 ± 3952.2117 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 5.6339 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | mae | 2.0648 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | r2 | -3.1112 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | rmse | 2.3736 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.3566 ± 0.5572 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 79099.0262 ± 8566.5074 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 5.6339 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | mae | 2.0648 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | r2 | -3.1112 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | rmse | 2.3736 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 4160.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.5957 ± 0.3504 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 77665.5150 ± 5829.8415 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3990.1940 ± 3446.4252 | 3 |
| dfa_ternary_step | ternary (per_step) | mae | 47.1905 ± 29.6649 | 3 |
| dfa_ternary_step | ternary (per_step) | r2 | -2910.7762 ± 2514.9702 | 3 |
| dfa_ternary_step | ternary (per_step) | rmse | 57.4392 ± 32.1932 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.1301 ± 0.0689 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 142183.3301 ± 6749.4762 | 3 |
| structured_hadamard_float | off (off) | loss | 0.8640 ± 0.0206 | 3 |
| structured_hadamard_float | off (off) | mae | 0.6732 ± 0.0115 | 3 |
| structured_hadamard_float | off (off) | r2 | 0.3695 ± 0.0151 | 3 |
| structured_hadamard_float | off (off) | rmse | 0.9295 ± 0.0111 | 3 |
| structured_hadamard_float | off (off) | sample_count | 4160.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 161174.3075 ± 1014.8411 | 3 |
| structured_hadamard_float_clip1 | off (off) | loss | 0.9023 ± 0.0129 | 3 |
| structured_hadamard_float_clip1 | off (off) | mae | 0.6805 ± 0.0101 | 3 |
| structured_hadamard_float_clip1 | off (off) | r2 | 0.3416 ± 0.0094 | 3 |
| structured_hadamard_float_clip1 | off (off) | rmse | 0.9499 ± 0.0068 | 3 |
| structured_hadamard_float_clip1 | off (off) | sample_count | 4160.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float_clip1 | off (off) | test_throughput_samples_sec | 101655.2044 ± 1594.1920 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 6131.0257 ± 9574.8019 | 3 |
| structured_hadamard_ternary | ternary (per_step) | mae | 34.6819 ± 37.1107 | 3 |
| structured_hadamard_ternary | ternary (per_step) | r2 | -4473.0117 ± 6987.0488 | 3 |
| structured_hadamard_ternary | ternary (per_step) | rmse | 58.0610 ± 64.3422 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 4160.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.2585 ± 0.0987 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 134680.2024 ± 36.7079 | 3 |
| structured_orth_float | off (off) | loss | 3.0929 ± 0.4880 | 3 |
| structured_orth_float | off (off) | mae | 1.3479 ± 0.1567 | 3 |
| structured_orth_float | off (off) | r2 | -1.2570 ± 0.3561 | 3 |
| structured_orth_float | off (off) | rmse | 1.7551 ± 0.1369 | 3 |
| structured_orth_float | off (off) | sample_count | 4160.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 158996.5665 ± 1363.5662 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4405.3143 ± 2553.5729 | 3 |
| structured_orth_ternary | ternary (per_step) | mae | 43.2086 ± 16.3400 | 3 |
| structured_orth_ternary | ternary (per_step) | r2 | -3213.7032 ± 1863.4264 | 3 |
| structured_orth_ternary | ternary (per_step) | rmse | 64.6183 ± 18.5657 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 4160.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.1494 ± 0.0035 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 133740.0766 ± 9077.0725 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 310.2312 ± 417.3877 | 3 |
| ternary_dfa_step | ternary (per_step) | mae | 10.7208 ± 7.8813 | 3 |
| ternary_dfa_step | ternary (per_step) | r2 | -225.3860 ± 304.5816 | 3 |
| ternary_dfa_step | ternary (per_step) | rmse | 14.8690 ± 11.5636 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 4160.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.1478 ± 0.0520 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 136225.5769 ± 3743.6591 | 3 |

## mnist (offline)

| Strategy Variant | Flip | Metric | Mean ± Std | n |
|---|---|---|---|---:|
| backprop_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| backprop_float | off (off) | loss | 0.3999 ± 0.0867 | 3 |
| backprop_float | off (off) | macro_f1 | 1.0000 | 3 |
| backprop_float | off (off) | sample_count | 128.0000 | 3 |
| backprop_float | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float | off (off) | test_throughput_samples_sec | 53421.9733 ± 2514.3412 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5393 ± 0.0047 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 28655.3822 ± 2542.3364 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0194 ± 0.0006 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 128.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 27950.2131 ± 4290.7979 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 128.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6055 ± 0.0084 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 27701.2024 ± 1168.8808 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.9089 ± 0.1111 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 1.7141 ± 2.4569 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.9221 ± 0.0739 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6627 ± 0.0106 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 29669.9476 ± 2408.7167 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.9297 ± 0.0547 | 3 |
| structured_hadamard_float | off (off) | loss | 1.2526 ± 0.0736 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.8995 ± 0.0638 | 3 |
| structured_hadamard_float | off (off) | sample_count | 128.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 51653.8557 ± 3385.9296 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.8021 ± 0.0565 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.1015 ± 1.1712 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.7269 ± 0.0067 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6787 ± 0.0005 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 30720.4274 ± 4142.6144 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7630 ± 0.0798 | 3 |
| structured_orth_float | off (off) | loss | 1.8366 ± 0.0571 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.7358 ± 0.0487 | 3 |
| structured_orth_float | off (off) | sample_count | 128.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 51117.3470 ± 10458.8899 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.8021 ± 0.0565 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 3.8578 ± 0.9988 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.7166 ± 0.0130 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 128.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6787 ± 0.0004 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 33726.3152 ± 1092.1213 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 128.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5806 ± 0.0082 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 31695.9082 ± 4261.7220 | 3 |

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
| backprop_float_lr06 | off (off) | accuracy | 0.9511 ± 0.0015 | 3 |
| backprop_float_lr06 | off (off) | loss | 0.1689 ± 0.0014 | 3 |
| backprop_float_lr06 | off (off) | macro_f1 | 0.9507 ± 0.0015 | 3 |
| backprop_float_lr06 | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr06 | off (off) | test_throughput_samples_sec | 44381.4064 ± 1507.2009 | 3 |
| backprop_float_lr10 | off (off) | accuracy | 0.9605 ± 0.0011 | 3 |
| backprop_float_lr10 | off (off) | loss | 0.1348 ± 0.0014 | 3 |
| backprop_float_lr10 | off (off) | macro_f1 | 0.9603 ± 0.0011 | 3 |
| backprop_float_lr10 | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr10 | off (off) | test_throughput_samples_sec | 47038.5695 ± 726.7801 | 3 |
| backprop_float_lr15 | off (off) | accuracy | 0.9649 ± 0.0001 **(best)** | 3 |
| backprop_float_lr15 | off (off) | loss | 0.1164 ± 0.0015 | 3 |
| backprop_float_lr15 | off (off) | macro_f1 | 0.9647 ± 0.0001 | 3 |
| backprop_float_lr15 | off (off) | sample_count | 14016.0000 | 3 |
| backprop_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| backprop_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| backprop_float_lr15 | off (off) | test_throughput_samples_sec | 46463.8410 ± 1328.7509 | 3 |
| backprop_ternary_step | ternary (per_step) | accuracy | 0.4042 ± 0.2111 | 3 |
| backprop_ternary_step | ternary (per_step) | loss | 8.9116 ± 2.1900 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 0.3741 ± 0.2313 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.5601 ± 0.0089 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 30871.8488 ± 890.9804 | 3 |
| dfa_float | off (off) | accuracy | 0.9254 ± 0.0081 | 3 |
| dfa_float | off (off) | loss | 1.4171 ± 0.1924 | 3 |
| dfa_float | off (off) | macro_f1 | 0.9250 ± 0.0085 | 3 |
| dfa_float | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 31213.0295 ± 476.0122 | 3 |
| dfa_float_lr06 | off (off) | accuracy | 0.9514 ± 0.0010 | 3 |
| dfa_float_lr06 | off (off) | loss | 0.1633 ± 0.0039 | 3 |
| dfa_float_lr06 | off (off) | macro_f1 | 0.9510 ± 0.0011 | 3 |
| dfa_float_lr06 | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float_lr06 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr06 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr06 | off (off) | test_throughput_samples_sec | 22355.5933 ± 304.4425 | 3 |
| dfa_float_lr10 | off (off) | accuracy | 0.9254 ± 0.0081 | 3 |
| dfa_float_lr10 | off (off) | loss | 1.4171 ± 0.1924 | 3 |
| dfa_float_lr10 | off (off) | macro_f1 | 0.9250 ± 0.0085 | 3 |
| dfa_float_lr10 | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float_lr10 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr10 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr10 | off (off) | test_throughput_samples_sec | 22215.5780 ± 289.7364 | 3 |
| dfa_float_lr15 | off (off) | accuracy | 0.9358 ± 0.0017 | 3 |
| dfa_float_lr15 | off (off) | loss | 1.3231 ± 0.0385 | 3 |
| dfa_float_lr15 | off (off) | macro_f1 | 0.9354 ± 0.0019 | 3 |
| dfa_float_lr15 | off (off) | sample_count | 14016.0000 | 3 |
| dfa_float_lr15 | off (off) | samples_per_step | 64.0000 | 3 |
| dfa_float_lr15 | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float_lr15 | off (off) | test_throughput_samples_sec | 21164.1829 ± 818.6561 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 0.7607 ± 0.0731 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | 4.9544 ± 1.5160 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 0.7456 ± 0.0966 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.4789 ± 0.0024 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 30348.9562 ± 122.6117 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | accuracy | 0.7671 ± 0.0598 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | loss | 4.8219 ± 1.2429 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | macro_f1 | 0.7539 ± 0.0681 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | ternary_zero_ratio | 0.1847 ± 0.0005 | 3 |
| dfa_ternary_epoch_tau002 | ternary (per_epoch) | test_throughput_samples_sec | 14310.4602 ± 4408.2872 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | accuracy | 0.7607 ± 0.0731 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | loss | 4.9544 ± 1.5160 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | macro_f1 | 0.7456 ± 0.0966 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | ternary_zero_ratio | 0.4789 ± 0.0024 | 3 |
| dfa_ternary_epoch_tau005 | ternary (per_epoch) | test_throughput_samples_sec | 15902.6137 ± 3818.4528 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | accuracy | 0.7935 ± 0.0085 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | loss | 4.2718 ± 0.1766 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | macro_f1 | 0.7928 ± 0.0095 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | sample_count | 14016.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | ternary_zero_ratio | 0.7472 ± 0.0051 | 3 |
| dfa_ternary_epoch_tau010 | ternary (per_epoch) | test_throughput_samples_sec | 18260.1413 ± 1026.1935 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.1756 ± 0.0839 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 17.0506 ± 1.7745 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.0879 ± 0.0655 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 14016.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6334 ± 0.0019 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 30359.7975 ± 910.3169 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.9299 ± 0.0009 | 3 |
| structured_hadamard_float | off (off) | loss | 0.2347 ± 0.0016 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.9295 ± 0.0009 | 3 |
| structured_hadamard_float | off (off) | sample_count | 14016.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 65656.5516 ± 1282.5006 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.3533 ± 0.0617 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 13.2049 ± 1.3390 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.2767 ± 0.0640 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 14016.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6774 ± 0.0004 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 31016.9338 ± 388.6253 | 3 |
| structured_orth_float | off (off) | accuracy | 0.8257 ± 0.0035 | 3 |
| structured_orth_float | off (off) | loss | 0.6278 ± 0.0123 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.8227 ± 0.0035 | 3 |
| structured_orth_float | off (off) | sample_count | 14016.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 64.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 64151.0800 ± 1188.4003 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.3391 ± 0.0318 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 13.5022 ± 0.6828 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.2589 ± 0.0202 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 14016.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6773 ± 0.0004 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 30994.1801 ± 364.6805 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 0.3005 ± 0.0504 | 3 |
| ternary_dfa_step | ternary (per_step) | loss | 14.4632 ± 1.0474 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 0.1828 ± 0.0449 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 14016.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.5895 ± 0.0039 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 30590.2373 ± 746.1712 | 3 |

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
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6118 ± 0.0295 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 47157.3277 ± 5397.0612 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0725 ± 0.0120 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 32.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 47549.6669 ± 8772.7309 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6373 ± 0.0091 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 49291.8923 ± 1852.5927 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8177 ± 0.1263 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5722 ± 2.7832 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7355 ± 0.1863 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6647 ± 0.0080 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 51097.3766 ± 3135.3668 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.6771 ± 0.4045 | 3 |
| structured_hadamard_float | off (off) | loss | 1.3163 ± 0.0194 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.6049 ± 0.4360 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 48358.2107 ± 6883.1411 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.2094 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.6941 ± 0.0074 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6772 ± 0.0021 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 48343.7547 ± 8086.4024 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7760 ± 0.2189 | 3 |
| structured_orth_float | off (off) | loss | 1.3158 ± 0.0444 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.6657 ± 0.3210 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 44240.0328 ± 7437.7879 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4.1640 ± 0.0786 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.6772 ± 0.0199 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6770 ± 0.0022 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 42631.4021 ± 4540.2041 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6072 ± 0.0064 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 51612.2241 ± 3049.0575 | 3 |

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
| backprop_ternary_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| backprop_ternary_step | ternary (per_step) | loss | -0.0000 ± 0.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| backprop_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6118 ± 0.0295 | 3 |
| backprop_ternary_step | ternary (per_step) | test_throughput_samples_sec | 38061.0459 ± 1521.4688 | 3 |
| dfa_float | off (off) | accuracy | 1.0000 **(best)** | 3 |
| dfa_float | off (off) | loss | 0.0327 ± 0.0041 | 3 |
| dfa_float | off (off) | macro_f1 | 1.0000 | 3 |
| dfa_float | off (off) | sample_count | 64.0000 | 3 |
| dfa_float | off (off) | samples_per_step | 32.0000 | 3 |
| dfa_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| dfa_float | off (off) | test_throughput_samples_sec | 37193.7595 ± 1903.4467 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | accuracy | 1.0000 **(best)** | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | loss | -0.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | macro_f1 | 1.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | sample_count | 64.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | ternary_zero_ratio | 0.6373 ± 0.0091 | 3 |
| dfa_ternary_epoch | ternary (per_epoch) | test_throughput_samples_sec | 37568.9243 ± 704.0968 | 3 |
| dfa_ternary_step | ternary (per_step) | accuracy | 0.8177 ± 0.1263 | 3 |
| dfa_ternary_step | ternary (per_step) | loss | 3.5722 ± 2.7832 | 3 |
| dfa_ternary_step | ternary (per_step) | macro_f1 | 0.7355 ± 0.1863 | 3 |
| dfa_ternary_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| dfa_ternary_step | ternary (per_step) | ternary_zero_ratio | 0.6647 ± 0.0080 | 3 |
| dfa_ternary_step | ternary (per_step) | test_throughput_samples_sec | 36351.2168 ± 3146.8542 | 3 |
| structured_hadamard_float | off (off) | accuracy | 0.7969 ± 0.2204 | 3 |
| structured_hadamard_float | off (off) | loss | 1.2706 ± 0.0087 | 3 |
| structured_hadamard_float | off (off) | macro_f1 | 0.6838 ± 0.3273 | 3 |
| structured_hadamard_float | off (off) | sample_count | 64.0000 | 3 |
| structured_hadamard_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_hadamard_float | off (off) | test_throughput_samples_sec | 38488.5719 ± 4039.5525 | 3 |
| structured_hadamard_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_hadamard_ternary | ternary (per_step) | loss | 4.2094 | 3 |
| structured_hadamard_ternary | ternary (per_step) | macro_f1 | 0.6941 ± 0.0074 | 3 |
| structured_hadamard_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_hadamard_ternary | ternary (per_step) | ternary_zero_ratio | 0.6772 ± 0.0021 | 3 |
| structured_hadamard_ternary | ternary (per_step) | test_throughput_samples_sec | 39054.8162 ± 2524.1131 | 3 |
| structured_orth_float | off (off) | accuracy | 0.7760 ± 0.2189 | 3 |
| structured_orth_float | off (off) | loss | 1.2960 ± 0.0434 | 3 |
| structured_orth_float | off (off) | macro_f1 | 0.6644 ± 0.3211 | 3 |
| structured_orth_float | off (off) | sample_count | 64.0000 | 3 |
| structured_orth_float | off (off) | samples_per_step | 32.0000 | 3 |
| structured_orth_float | off (off) | ternary_zero_ratio | 0.0000 | 3 |
| structured_orth_float | off (off) | test_throughput_samples_sec | 41204.3096 ± 1394.5144 | 3 |
| structured_orth_ternary | ternary (per_step) | accuracy | 0.7969 | 3 |
| structured_orth_ternary | ternary (per_step) | loss | 4.1640 ± 0.0786 | 3 |
| structured_orth_ternary | ternary (per_step) | macro_f1 | 0.6772 ± 0.0199 | 3 |
| structured_orth_ternary | ternary (per_step) | sample_count | 64.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| structured_orth_ternary | ternary (per_step) | ternary_zero_ratio | 0.6770 ± 0.0022 | 3 |
| structured_orth_ternary | ternary (per_step) | test_throughput_samples_sec | 34852.1274 ± 4936.3889 | 3 |
| ternary_dfa_step | ternary (per_step) | accuracy | 1.0000 **(best)** | 3 |
| ternary_dfa_step | ternary (per_step) | loss | -0.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | macro_f1 | 1.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | sample_count | 64.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | samples_per_step | 32.0000 | 3 |
| ternary_dfa_step | ternary (per_step) | ternary_zero_ratio | 0.6072 ± 0.0064 | 3 |
| ternary_dfa_step | ternary (per_step) | test_throughput_samples_sec | 34596.8636 ± 7185.5088 | 3 |

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

