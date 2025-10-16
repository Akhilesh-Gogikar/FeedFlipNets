"""Core numerical primitives for FeedFlipNets.

Backwards-compat note: the legacy module ``feedflipnets.core.feedback`` remains
available for direct import, but is no longer imported here to avoid triggering
its deprecation warning during package import. Prefer ``feedflipnets.core.strategies``.
"""

from . import activations, quant, strategies, types

__all__ = ["activations", "quant", "strategies", "types"]
