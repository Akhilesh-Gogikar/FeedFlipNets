# Project status

- **Final state:** Closed
- **Owner:** None
- **Closeout date:** 2026-07-11
- **Canonical source commit:** `912485d0f00612aff0e3049ac888c3b47c454204`
- **Canonical branch:** `archive/feedflip-closed-2026-07-11`

## Evidence and conclusion

Five frozen, preregistered rounds answered the intended bottleneck question on the recorded MLP benchmark. Round five's exact-gradient 8-bit oracle reached `0.609`, above the `0.604` anchor, so 8-bit state precision is not binding. The transport-free gate remained NO-GO at `0.554 < 0.60`; the residual gap is attributed to the update/feedback rule rather than accumulator precision. This closes the scoped FeedFlip question without claiming a universal impossibility result.

Focused verification of the preserved code-review line completed on 2026-07-11: dataset loaders, lock-free probe, and transport-free LM tests — **10 passed**, with two non-failing warnings.

## Preserved refs

- `archive/feedflip-closed-2026-07-11` contains all six commits that were ahead of `origin/main` plus this record.
- `archive/feedflip-code-review-2026-07-11` at `79517589abd0f0bb2101173a513ec90f8c9d9db9` preserves the unmerged review fixes and its prior dirty repository maps.
- The round-five result remains reachable at `59ace09911c926d73a8c32c2184878dc477d50c1`.
- `origin/main` was observed at `81a6e35c226c96c2b5ba54ef0ade3bba479fa68c` before closeout.

## Risks

The result is benchmark- and mechanism-specific. The code-review branch has not been reconciled with the later round-four/round-five history and must not be merged automatically.

## Resume condition

Resume only with a named owner, a materially new falsifiable question, and a preregistered gate that would change the conclusion.

## First step if resumed

Reproduce round five from the pinned result/configuration, then replay the code-review branch in a disposable worktree and rerun its focused tests.
