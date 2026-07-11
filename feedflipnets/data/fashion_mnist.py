"""Fashion-MNIST dataset with deterministic splits and offline fixtures.

This mirrors the MNIST loader but sources data from OpenML when
``offline=False`` to avoid heavyweight dependencies. In offline mode we
provide a deterministic synthetic fixture that preserves the input
dimensionality and class structure for quick, reproducible runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.datasets import fetch_openml

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir


def _offline_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic Fashion-MNIST–like synthetic dataset.

    We generate 10 classes with simple localized patterns plus noise to
    emulate class separation while remaining extremely lightweight.
    """

    rng = np.random.default_rng(2468)
    num_classes = 10
    samples_per_class = 64
    height, width = 28, 28
    flat_dim = height * width

    base_patterns: list[np.ndarray] = []
    # Build simple localized blobs per class
    for cls in range(num_classes):
        img = np.zeros((height, width), dtype=np.float32)
        r0 = (cls * 3) % (height - 6)
        c0 = (cls * 5) % (width - 6)
        img[r0 : r0 + 6, c0 : c0 + 6] = 1.0
        base_patterns.append(img.reshape(-1))

    images: list[np.ndarray] = []
    labels: list[int] = []
    for cls, base in enumerate(base_patterns):
        for _ in range(samples_per_class):
            noise = rng.normal(loc=0.0, scale=0.10, size=flat_dim).astype(np.float32)
            sample = np.clip(base + noise, 0.0, 1.0)
            images.append(sample.reshape(height, width))
            labels.append(cls)

    X = np.stack(images, axis=0).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def _prepare_inputs(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32)
    if images.max() > 1:
        images /= 255.0
    return images.reshape(images.shape[0], -1).astype(np.float32)


def _prepare_targets(labels: np.ndarray, *, one_hot: bool, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64)
    if one_hot:
        eye = np.eye(num_classes, dtype=np.float32)
        return eye[labels]
    return labels.astype(np.float32).reshape(-1, 1)


@register_dataset("fashion_mnist")
def build_fashion_mnist(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for Fashion-MNIST.

    In online mode, data is fetched from OpenML (dataset name
    "Fashion-MNIST"). In offline mode, a deterministic synthetic fixture
    is used.
    """

    resolve_cache_dir(cache_dir)  # ensure directory exists
    if offline:
        inputs_raw, labels_raw = _offline_dataset()
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        # OpenML returns float64 arrays by default – cast to float32
        ds = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
        inputs_raw = ds.data.astype(np.float32)
        # targets are strings; cast to int then to np.int64
        labels_raw = ds.target.astype(np.int64)
        provenance = {
            "mode": "download",
            "openml_name": "Fashion-MNIST",
            "n_samples": int(inputs_raw.shape[0]),
        }

    inputs = _prepare_inputs(inputs_raw)
    num_classes = 10
    targets = _prepare_targets(labels_raw, one_hot=one_hot, num_classes=num_classes)

    splits = deterministic_split(
        inputs.shape[0], val_split=val_split, test_split=test_split, seed=seed
    )

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(
            inputs,
            targets,
            indices,
            batch_size=batch_size,
            seed=split_seed,
            replacement=(split == "train"),
        )

    data_spec = DataSpec(
        d_in=int(inputs.shape[1]),
        d_out=(num_classes if one_hot else 1),
        task_type="multiclass",
        num_classes=num_classes,
        normalization={"inputs": {"method": "minmax", "range": [0.0, 1.0]}},
        extra={"input_shape": (28, 28)},
    )

    provenance.update(
        {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "one_hot": one_hot,
        }
    )

    split_sizes = splits.sizes

    return DatasetSpec(
        name="fashion_mnist",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in split_sizes.items()},
    )


__all__ = ["build_fashion_mnist"]
