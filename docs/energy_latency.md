Energy and Latency Measurement Guide
====================================

Scope
-----
- Goal: measure latency and energy/inference using accepted yardsticks for edge/MCU devices.
- Recommended: MLPerf Tiny (MLCommons) and EEMBC MLMark.

What we report here
-------------------
- Host CPU latency and peak RAM for one representative model are reported in README and the paper’s Deployability table.
- Energy/inference is not included in this repo’s artifacts yet; use the methods below to add it.

Method A: MLPerf Tiny
---------------------
1. Select target hardware (e.g., Cortex‑M4/M7, RISC‑V MCU) with power measurement support.
2. Install MLPerf Tiny per MLCommons instructions (TFLM/CMSIS‑NN or vendor stack).
3. Port a representative fully‑connected model approximating 784‑256‑256‑10 ternary forward weights; keep float “shadow” off for inference.
4. Integrate a power measurement harness (board’s shunt, DAQ, or vendor tooling) and calibrate sampling frequency.
5. Run inference in single‑sample mode; report:
   - Accuracy for the chosen dataset/task
   - Latency (median, p95)
   - Energy/inference (median, p95) in mJ
   - Device, compiler, clock, voltage, ambient

Method B: EEMBC MLMark
----------------------
1. Use the MLMark SDK for your MCU/board.
2. Implement or adapt a dense layer network matching the FC topology with ternary weights converted to int2/int8 storage as supported.
3. Configure the MLMark workload; enable energy measurement if supported on your board.
4. Capture latency and energy metrics; report with device profile as above.

Practical tips
--------------
- Quantization: store ternary weights packed (2 bits) or map to int8 {‑1,0,+1} for portability.
- Activations: use int8/int16 where supported; otherwise float32.
- Determinism: fix seeds and input sets; report firmware and toolchain versions.
- Statistics: report median and p95 across N≥100 runs; include error bars if available.

Minimum bar for papers
----------------------
- One credible device/profile with accuracy–latency–energy trade‑offs earns more credibility than generic “edge‑ready” claims.
- If energy measurement is unavailable, report wall‑clock latency and memory and clearly state limitations.

