# FeedFlipNets Modality Results (offline fixtures)

All runs were executed offline with deterministic seeds. Test-set metrics shown.


| Preset | Modality | Dataset | Flip | Schedule | Epochs | Acc | Macro-F1 | MAE | RMSE | R2 | Samples/Step | Test Throughput (samples/s) | Zero Ratio | Run Dir |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mnist_mlp_dfa | vision | mnist | ternary | per_step | 5 | 0.9688 | 0.8857 |  |  |  | 64.00 |  | 0.673 | runs/mnist-mlp-dfa |
| ucr_gunpoint_mlp_dfa | time_series | ucr | ternary | per_step | 6 | 1.0000 | 1.0000 |  |  |  | 32.00 |  | 0.652 | runs/ucr-gunpoint-mlp-dfa |
| california_housing_mlp_dfa | tabular | california_housing | off |  | 8 |  |  | 1.8604 | 2.4205 | 0.0731 | 64.00 |  | 0.000 | runs/california-housing-mlp-dfa |
| 20newsgroups_bow_mlp_dfa | text | 20newsgroups | ternary | per_step | 6 | 0.2188 | 0.1467 |  |  |  | 64.00 |  | 0.682 | runs/20newsgroups-bow-mlp-dfa |
