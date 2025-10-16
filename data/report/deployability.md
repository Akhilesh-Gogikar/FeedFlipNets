# Deployability (host profile)

Device: Apple M3 (Darwin 24.6.0, arm64)

| Model | Params | Bit-width | Sparsity by layer | Peak RAM (KB) | CPU Latency (ms) | Energy/inf. |
|---|---:|:---:|:---|---:|---:|:---|
| MNIST MLP (784-256-256-10) | 268800 | 2 (ternary) | 0.68, 0.68, 0.67 | 68.7 | 0.012 (p95 0.019) | N/A (MLPerf Tiny/EEMBC) |
| 20NG BoW MLP (2048-512-256-20) | 1184768 | 2 (ternary) | 0.68, 0.68, 0.68 | 297.2 | 0.029 (p95 0.295) | N/A (MLPerf Tiny/EEMBC) |
| AG News TFIDF MLP (2048-512-128-4) | 1114624 | 2 (ternary) | 0.68, 0.69, 0.72 | 280.1 | 0.026 (p95 0.041) | N/A (MLPerf Tiny/EEMBC) |
| UCR GunPoint MLP (150-128-128-2) | 35840 | 2 (ternary) | 0.68, 0.68, 0.70 | 9.3 | 0.008 (p95 0.008) | N/A (MLPerf Tiny/EEMBC) |

Notes: weights bit-width excludes activations; peak RAM uses packed ternary for weights and float32 activations; latency uses NumPy forward (BLAS).
latency uses NumPy forward (BLAS).
