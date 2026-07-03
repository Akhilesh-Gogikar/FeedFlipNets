MCU Integration Templates
=========================

This folder outlines simple ways to use exported ternary weights on microcontrollers.

Quick start (header-only)
-------------------------
1. Export weights from a trained checkpoint:
   - python3 scripts/export_ternary.py --ckpt runs/.../best.ckpt --name my_model --format c
2. Drop the generated header into your firmware:
   - artifacts/export/my_model_ternary.h
3. Implement a dense layer using int8 activations and ternary weights {-1,0,+1}:
   - y[j] = sum_i x[i] * w[i,j], with w in {-1,0,+1}
   - Use branchless tricks for speed: acc += x[i] * (w==1) - x[i] * (w==-1)

Binary format (packed 2-bit)
----------------------------
- Use artifacts/export/my_model_ternary_packed.bin and my_model_manifest.json.
- Mapping: code 0 -> 0, 1 -> +1, 2 -> -1, 3 -> unused.
- Unpack 4 weights per byte: code = (byte >> (2*k)) & 0x3.

MLPerf Tiny / EEMBC MLMark
--------------------------
- See docs/energy_latency.md for methodology and checklist.
- Suggested path: integrate the header or packed weights into an FC-only workload and measure latency/energy on your target MCU.

