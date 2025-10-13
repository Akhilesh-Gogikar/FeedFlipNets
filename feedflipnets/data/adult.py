"""UCI Adult dataset (binary classification) with deterministic fallback.

Online mode downloads the Adult dataset via OpenML and applies a minimal
one-hot encoding using pandas. Offline mode returns a small synthetic,
linearly separable fixture for fast, deterministic tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir, standardize


def _offline_dataset(
    n: int = 512, d: int = 16, *, seed: int = 2025
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic, linearly separable binary classification fixture."""

    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, d)).astype(np.float32)
    w = rng.normal(0, 1, size=(d, 1)).astype(np.float32)
    logits = X @ w + 0.25 * rng.normal(0, 1, size=(n, 1)).astype(np.float32)
    y = (logits > 0).astype(np.int64).reshape(-1)
    return X.astype(np.float32), y


def _prepare_targets(labels: np.ndarray, *, one_hot: bool) -> np.ndarray:
    labels = labels.astype(np.int64)
    if one_hot:
        eye = np.eye(2, dtype=np.float32)
        return eye[labels]
    return labels.astype(np.float32).reshape(-1, 1)


@register_dataset("adult")
def build_adult(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = False,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for the UCI Adult dataset.

    Targets are binary (<=50K vs >50K). If ``one_hot`` is True, targets are
    2D one-hot; otherwise, a single 0/1 column is returned.
    """

    resolve_cache_dir(cache_dir)
    if offline:
        X, y_idx = _offline_dataset(seed=seed)
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        ds = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        # OpenML returns: data (DataFrame), target (Series or ndarray)
        X_cat = ds.data  # type: ignore[assignment]
        if X_cat is None:
            raise RuntimeError("OpenML adult dataset returned no data frame")
        y_series = pd.Series(ds.target)  # ensure Series
        # Target is string/categorical; map to {0,1}
        y_raw = (y_series.astype(str) == ">50K").astype(int).to_numpy().astype(np.int64)
        X_df = pd.get_dummies(X_cat, drop_first=True)
        X = X_df.to_numpy(dtype=np.float32, copy=True)
        # Standardize features for stability
        X, _, _ = standardize(X)
        y_idx = y_raw
        provenance = {"mode": "download", "openml_id": int(getattr(ds, "data_id", 0))}

    targets = _prepare_targets(y_idx, one_hot=one_hot)
    splits = deterministic_split(X.shape[0], val_split=val_split, test_split=test_split, seed=seed)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(X, targets, indices, batch_size=batch_size, seed=split_seed)

    data_spec = DataSpec(
        d_in=int(X.shape[1]),
        d_out=(2 if one_hot else 1),
        task_type="binary",
        num_classes=2,
        normalization={"inputs": {"method": "standardize"}},
    )

    provenance.update(
        {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "one_hot": one_hot,
            "n_features": int(X.shape[1]),
            "n_samples": int(X.shape[0]),
        }
    )

    return DatasetSpec(
        name="adult",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["build_adult"]
