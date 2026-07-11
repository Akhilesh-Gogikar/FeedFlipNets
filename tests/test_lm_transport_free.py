# tests/test_lm_transport_free.py
"""GUARD 2 — ②'s R_O adaptation is TRANSPORT-FREE: no autograd, no W_Oᵀ.

②'s node-perturbation of Ctx (ghat_ctx_nodepert) and the R_O Kolen-Pollack update must NEVER call
torch.autograd.grad and NEVER dereference the downstream transpose W_Oᵀ (a FORWARD P["Wo"] matmul is
allowed). Since the ② path is pure NumPy there is no autograd graph to taint-trace, so the
authoritative checks are (i) structural source-inspection, (ii) a positive control that DELIBERATELY
uses transport (reads Wo.T) which the checker MUST flag — without a flagged positive control the
check is worthless (a real M2 finding), and (iii) an e-fixed transpose-perturbation control:
perturbing Wo (which would move Wo.T if it were read) leaves ② changing only through the forward Wo
it legitimately uses, while the transport control moves via the transpose it reads.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ast  # noqa: E402
import inspect  # noqa: E402
import textwrap  # noqa: E402

import numpy as np  # noqa: E402

from feedflipnets.core import lm as lmmod  # noqa: E402
from feedflipnets.core.lm import block_forward_np, ghat_ctx_nodepert  # noqa: E402


def _reads_transport(fn) -> bool:
    """Structural source-inspection: True iff fn's CODE transposes/dereferences the downstream
    weight Wo (a W_Oᵀ read) or calls torch.autograd. Authoritative lock-free check for the
    pure-NumPy ② adaptation path.

    The check parses the AST so it inspects what the code DOES, not what a docstring SAYS: the ②
    docstring legitimately states "never calls autograd; never reads W_Oᵀ" and that prose must not
    trip the checker. Flags:
      - any attribute access named `.T` whose value references `Wo` (e.g. P["Wo"].T, block.Wo.T)  →
        the forbidden transpose transport.
      - any attribute access under the `autograd` namespace (e.g. torch.autograd.grad).
    A FORWARD P["Wo"] matmul (no trailing .T) is NOT flagged — that is the legitimate ② read.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))

    def _mentions_wo(node) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr == "Wo":
                return True  # e.g. block.Wo, self.Wo
            if isinstance(sub, ast.Constant) and sub.value == "Wo":
                return True  # e.g. P["Wo"]
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr == "autograd":
                return True
            if node.attr == "T" and _mentions_wo(node.value):
                return True
    return False


def test_ghat_is_transport_free():
    # ② adaptation reads P["Wo"] FORWARD only; never W_Oᵀ, never autograd.
    assert _reads_transport(ghat_ctx_nodepert) is False


def test_train_step_adaptation_has_no_autograd_or_transpose():
    # AST-level (code, not prose): train_step's ② branch never calls autograd nor reads Wo.T, even
    # though its comment legitimately says "NO autograd / NO W_Oᵀ".
    assert _reads_transport(lmmod.NumpyLM.train_step) is False


def test_positive_control_is_flagged():
    # A transport-USING estimator (reads Wo.T) MUST be flagged — proves the check CAN fail.
    def ghat_transport_control(P, c, e_top_block, rho, K_samp, rng):
        # forbidden: dL/dCtx via the actual transpose W_Oᵀ (exactly what ② must avoid)
        return e_top_block @ P["Wo"].T

    assert _reads_transport(ghat_transport_control) is True


def test_bp_style_transpose_reader_is_flagged():
    # A bp-style grad that reads the transpose is also flagged — a second positive control.
    def bp_style_dctx(P, c, e_top_block):
        return e_top_block @ P["Wo"].T  # W_Oᵀ transport — the thing ② is forbidden to do

    assert _reads_transport(bp_style_dctx) is True


def test_efixed_transpose_perturbation_leaves_two_invariant_but_moves_transport():
    """e + forward cache fixed: perturb Wo by a constant offset.

    ② (ghat_ctx_nodepert) reads Wo FORWARD only, so with rng and the perturbation ξ held fixed its
    dependence on Wo is exactly the forward matmul it legitimately uses — it does NOT read Wo.T. The
    positive control reads Wo.T explicitly, so the SAME Wo perturbation moves it through the
    transpose. This separates 'legitimate forward reader' from 'transport reader'.
    """
    rng = np.random.default_rng(0)
    d, h, T, B = 8, 16, 6, 4
    P = dict(
        Wq=rng.standard_normal((d, d)),
        Wk=rng.standard_normal((d, d)),
        Wv=rng.standard_normal((d, d)),
        Wo=rng.standard_normal((d, d)),
        W1=rng.standard_normal((d, h)),
        W2=rng.standard_normal((h, d)),
    )
    x = rng.standard_normal((B, T, d))
    e_top = rng.standard_normal((B, T, d))

    # ② is a well-formed FORWARD-only estimator: finite, non-trivial output.
    c = block_forward_np(P, x)
    g0 = ghat_ctx_nodepert(P, c, e_top, rho=0.02, K_samp=8, rng=np.random.default_rng(1))
    assert np.isfinite(g0).all() and np.linalg.norm(g0) > 0

    # The forbidden transport control depends on Wo.T explicitly; perturbing Wo moves it via the
    # transpose it reads.
    ctrl0 = e_top @ P["Wo"].T
    P2 = dict(P)
    P2["Wo"] = P["Wo"] + 5.0
    ctrl1 = e_top @ P2["Wo"].T
    assert np.max(np.abs(ctrl1 - ctrl0)) > 1.0  # the forbidden path is transpose-sensitive

    # ② DOES move under the Wo perturbation — but only through the FORWARD matmul it
    # legitimately uses (Ctx_pert @ P["Wo"] inside its local-loss recompute). Exact
    # invariance to Wo would be wrong for a forward reader; transpose-freedom is pinned
    # by the AST checks above. Execute the moved half so it isn't vacuous:
    g1 = ghat_ctx_nodepert(P2, c, e_top, rho=0.02, K_samp=8, rng=np.random.default_rng(1))
    assert not np.allclose(g1, g0)

    # The executable INVARIANT half: ② never reads Wq/Wk/Wv (with the cache c fixed),
    # so perturbing them must leave the estimate bit-identical (same rng seed).
    P3 = dict(P)
    P3["Wq"] = P["Wq"] + 5.0
    P3["Wk"] = P["Wk"] + 5.0
    P3["Wv"] = P["Wv"] + 5.0
    g2 = ghat_ctx_nodepert(P3, c, e_top, rho=0.02, K_samp=8, rng=np.random.default_rng(1))
    assert np.allclose(g2, g0)
