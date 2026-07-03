"""Analytical pipeline cost model: DFA-pipeline vs tuned 1F1B(+checkpointing).

Falsification for the speed thesis: is there a realistic (S, m, tu, memory) operating point
where DFA-pipeline throughput / best-tuned-BP >= 1.5x? Normalize forward tf = 1. First-cut
analytical model (not simulator-validated) — see data/report/speed/README.md.

Schedules (m microbatches, S stages):
- GPipe BP:  T = (m+S-1)*(tf + tb_eff);  peak activation mem ~ m * a  (all stashed for backward).
- 1F1B  BP:  T = (m+S-1)*(tf + tb_eff);  peak activation mem ~ S * a  (bounded in-flight).
             tb_eff = tb + (tf if must_checkpoint else 0)   # checkpointing recomputes forward.
- DFA:       T = m*(tf + tu) + (S-1)*tf;  peak activation mem ~ 1 * a  (freed after local update).
             No backward dependency chain -> no backward-fill bubble; no checkpointing needed.

BP is "tuned": 1F1B, with checkpointing enabled only when needed to fit the memory budget.
"""

TF = 1.0
R_B = 2.0  # backward cost / forward cost (BP)


def bp_time_mem(S, m, a, mem_budget):
    # 1F1B: bounded activation memory ~ S*a; checkpoint if it doesn't fit (recomputes forward).
    mem = S * a
    must_ckpt = mem > mem_budget
    if must_ckpt:
        mem = 1 * a  # checkpointing keeps only the stage-boundary activation
    tb_eff = R_B * TF + (TF if must_ckpt else 0.0)
    T = (m + S - 1) * (TF + tb_eff)
    return T, mem, must_ckpt


def dfa_time_mem(S, m, a, tu):
    T = m * (TF + tu) + (S - 1) * TF
    mem = 1 * a
    return T, mem


def throughput(T, m):
    return m / T


def scan():
    a = 1.0
    best = {"ratio": 0.0}
    hits = []
    for S in [4, 8, 16, 32]:
        for mult in [1, 2, 4, 8]:
            m = mult * S
            for tu in [0.5, 1.0, 1.5, 2.0]:
                for mem_budget in [2 * a, S * a, 1e9]:  # tight / just-fits-1F1B / unlimited
                    T_bp, mem_bp, ckpt = bp_time_mem(S, m, a, mem_budget)
                    T_dfa, mem_dfa = dfa_time_mem(S, m, a, tu)
                    r = throughput(T_dfa, m) / throughput(T_bp, m)
                    rec = {
                        "S": S,
                        "m": m,
                        "tu": tu,
                        "mem": mem_budget,
                        "ckpt": ckpt,
                        "ratio": round(r, 3),
                        "mem_edge": round(mem_bp / mem_dfa, 1),
                    }
                    if r > best["ratio"]:
                        best = rec
                    if r >= 1.5:
                        hits.append(rec)
    return best, hits


def main():
    best, hits = scan()
    print("=== DFA-pipeline throughput / tuned-1F1B, scan over (S, m, tu, mem) ===")
    print(f"BEST operating point: {best}")
    print(f"\n# operating points with ratio >= 1.5x: {len(hits)}")
    print("\nSTANDARD well-tuned point m=4S, mem=just-fits (no forced checkpoint):")
    for S in [8, 16, 32]:
        for tu in [0.5, 1.0, 1.5, 2.0]:
            T_bp, _, ckpt = bp_time_mem(S, 4 * S, 1.0, S * 1.0)
            T_dfa, _ = dfa_time_mem(S, 4 * S, 1.0, tu)
            r = throughput(T_dfa, 4 * S) / throughput(T_bp, 4 * S)
            print(f"  S={S:2d} tu={tu}: ratio={r:.3f}  (ckpt={ckpt})")
    print("\nTIGHT memory budget (BP forced to checkpoint, DFA not), m=4S:")
    for S in [8, 16, 32]:
        for tu in [0.5, 1.0]:
            T_bp, _, ckpt = bp_time_mem(S, 4 * S, 1.0, 2.0)
            T_dfa, _ = dfa_time_mem(S, 4 * S, 1.0, tu)
            r = throughput(T_dfa, 4 * S) / throughput(T_bp, 4 * S)
            print(f"  S={S:2d} tu={tu}: ratio={r:.3f}  (BP ckpt={ckpt})")
    print("\nNOTE: ignores the e-broadcast comm (|e|*S, large at real vocab) and assumes")
    print("DFA updates overlap forward; both would REDUCE the ratio. Hinges on tu (unmeasured).")


if __name__ == "__main__":
    main()
