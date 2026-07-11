"""MNIST dataset with deterministic splits and offline fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.types import Batch
from .cache import fetch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import SplitIndices, batch_iterator, deterministic_split, resolve_cache_dir

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_CHECKSUM = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


def _load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    data = np.load(path)

    def _lookup(*keys: str) -> np.ndarray:
        for key in keys:
            if key in data:
                return data[key]
        raise KeyError(f"None of {keys} found in MNIST archive")

    x_train = _lookup("X_train", "x_train").astype(np.float32)
    y_train = _lookup("y_train", "Y_train").astype(np.int64)
    x_test = _lookup("X_test", "x_test").astype(np.float32)
    y_test = _lookup("y_test", "Y_test").astype(np.int64)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return x, y, int(x_train.shape[0])


def _prepare_inputs(images: np.ndarray) -> np.ndarray:
    images = images.astype(np.float32)
    if images.max() > 1:
        images /= 255.0
    images = images.reshape(images.shape[0], -1)
    return images.astype(np.float32)


def _prepare_targets(labels: np.ndarray, *, one_hot: bool, num_classes: int) -> np.ndarray:
    labels = labels.astype(np.int64)
    if one_hot:
        eye = np.eye(num_classes, dtype=np.float32)
        return eye[labels]
    return labels.astype(np.float32).reshape(-1, 1)


def _offline_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic, linearly-separable MNIST-style dataset."""

    rng = np.random.default_rng(12345)
    num_classes = 10
    samples_per_class = 64
    height, width = 28, 28
    flat_dim = height * width
    segment = max(1, flat_dim // num_classes)

    images: list[np.ndarray] = []
    labels: list[int] = []

    for cls in range(num_classes):
        base = np.zeros(flat_dim, dtype=np.float32)
        start = cls * segment
        end = start + segment
        if end <= flat_dim:
            base[start:end] = 1.0
        else:
            base[start:] = 1.0
            base[: end - flat_dim] = 1.0

        for _ in range(samples_per_class):
            noise = rng.normal(loc=0.0, scale=0.05, size=flat_dim).astype(np.float32)
            sample = np.clip(base + noise, 0.0, 1.0)
            images.append(sample.reshape(height, width))
            labels.append(cls)

    images_arr = np.stack(images, axis=0)
    labels_arr = np.asarray(labels, dtype=np.int64)
    return images_arr, labels_arr


def _canonical_split(n_samples: int, n_train: int, *, val_split: float, seed: int) -> SplitIndices:
    """Preserve the official test set; carve val from the train portion only."""

    rng = np.random.default_rng(seed)
    train_indices = np.arange(n_train)
    rng.shuffle(train_indices)
    val_size = int(round(n_train * val_split))
    val_size = min(max(val_size, 1 if val_split > 0 else 0), n_train - 1)
    return SplitIndices(
        train=train_indices[val_size:],
        val=train_indices[:val_size],
        test=np.arange(n_train, n_samples),
    )


@register_dataset("mnist")
def build_mnist(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for MNIST."""

    cache_root = resolve_cache_dir(cache_dir)
    n_train_official: int | None = None
    if offline:
        inputs_raw, labels_raw = _offline_dataset()
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        path, provenance = fetch(
            name="mnist",
            url=MNIST_URL,
            checksum=MNIST_CHECKSUM,
            filename="mnist.npz",
            offline_path=None,
            offline_builder=None,
            offline=False,
            cache_dir=cache_root,
        )

        inputs_raw, labels_raw, n_train_official = _load_archive(path)
    inputs = _prepare_inputs(inputs_raw)
    targets = _prepare_targets(labels_raw, one_hot=one_hot, num_classes=10)

    if n_train_official is not None:
        # Preserve the canonical MNIST test set; carve val from official train.
        splits = _canonical_split(inputs.shape[0], n_train_official, val_split=val_split, seed=seed)
    else:
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

    target_dim = 10 if one_hot else 1
    data_spec = DataSpec(
        d_in=int(inputs.shape[1]),
        d_out=target_dim,
        task_type="multiclass",
        num_classes=10,
        normalization={"inputs": {"method": "minmax", "range": [0.0, 1.0]}},
        extra={"input_shape": (28, 28)},
    )

    provenance = dict(provenance)
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
        name="mnist",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in split_sizes.items()},
    )


__all__ = ["build_mnist"]
