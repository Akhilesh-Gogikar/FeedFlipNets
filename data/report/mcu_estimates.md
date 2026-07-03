# MCU Estimates (ops-based)

Assumptions: cycles_per_mac=1.0, clock=100.0 MHz; nonzero-weight MACs only.

| Model | Nonzero MACs | Cycles (est.) | Latency (ms, est.) |
|---|---:|---:|---:|
| MNIST MLP (784-256-256-10) | 85311 | 85311 | 0.853 |
| 20NG BoW MLP (2048-512-256-20) | 375905 | 375905 | 3.759 |
| AG News TFIDF MLP (2048-512-128-4) | 353678 | 353678 | 3.537 |
| UCR GunPoint MLP (150-128-128-2) | 11296 | 11296 | 0.113 |
