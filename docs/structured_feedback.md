Structured Feedback Matrices (B_l)

- Shapes: For a network with layer sizes `dims = [d0, d1, ..., dL]` we use one feedback matrix per hidden layer `l = 1..L-1` with shape `(dL, dl)`. This projects the output-layer error directly to layer `l` as in classical DFA.

- Seeding: Each `B_l` is generated with its own derived seed recorded in `StrategyState.metadata['layer_seeds']` along with `layer_shapes`. This ensures reproducible regeneration independent of the global RNG state.

- Options and generation rules:
  - `orthogonal` (Haar/QR):
    - If `dL <= dl`, draw `A ~ N(0,1)^{dl x dL}`, compute QR in reduced mode, apply the standard sign correction to columns (Mezzadri/Higham), and return `Q^T` of shape `(dL x dl)`. Rows are orthonormal, so `B_l B_l^T = I_{dL}` and `||δ_L B_l||_2 = ||δ_L||_2`.
    - If `dL > dl`, draw `A ~ N(0,1)^{dL x dl}`, compute reduced QR with sign correction, and return `Q` of shape `(dL x dl)` with orthonormal columns. This mapping is non‑expansive (`||⋅ B_l||_2 <= ||⋅||_2`).
    - References: Mezzadri (2006), Higham (blog) for QR + Haar details.
  - `hadamard` (SRHT-style):
    - Let `s = 2^ceil(log2(dl))`. Build the size‑`s` normalized Hadamard `H_s` (entries ±1/√s). Sample `dL` rows and `dl` columns uniformly without replacement and apply random column signs. When `s > dl`, multiply by `√(s/dl)` to debias cropping so expected row norms are 1. Returns shape `(dL x dl)`.
    - This yields a near‑isometry: preserves norms in expectation with JL/FJLT concentration. Fast transforms and ±1 structure are hardware friendly.
    - References: Ailon & Chazelle (2009) for FJLT.

- Non‑square scaling rules:
  - No extra scaling is needed for Haar/QR: rows (or columns) are exactly orthonormal, keeping operator norm at 1.
  - For SRHT when `s > dl`, apply `√(s/dl)` to correct for column cropping; this keeps expected row norms at 1 and stabilizes magnitudes across layers.

Why this helps

Near‑isometric projections keep error energy stable across layers: row‑orthonormal `B_l` guarantee `||δ_L B_l||_2 = ||δ_L||_2`, while SRHT approximates this with high probability. Stable energy improves layer‑wise conditioning and reduces variance in the DFA update direction. Empirically, this lifts alignment curves (higher cosine between DFA and BP updates) and reduces exploding/vanishing behavior in deeper or narrower networks.

Pointers

- Francesco Mezzadri (2006). How to generate random matrices from the classical compact groups.
- Nicholas J. Higham (2015). Random orthogonal matrices and the QR factorization (blog).
- Nir Ailon, Bernard Chazelle (2009). The Fast Johnson–Lindenstrauss Transform and Approximate Nearest Neighbors.

