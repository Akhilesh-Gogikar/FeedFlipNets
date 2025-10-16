"""CIFAR-10 dataset with deterministic splits and offline fixtures.

Offline mode provides a tiny, deterministic synthetic fixture preserving
input dimensionality (32x32x3 -> 3072) and class structure. Online mode
optionally uses torchvision if available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir


def _offline_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic CIFAR‑like synthetic dataset.

    We generate 10 classes with small colored patches at class‑dependent
    locations, plus Gaussian noise. Images are in [0,1].
    """

    rng = np.random.default_rng(13579)
    num_classes = 10
    samples_per_class = 64
    H, W, C = 32, 32, 3
    flat_dim = H * W * C

    images: list[np.ndarray] = []
    labels: list[int] = []
    for cls in range(num_classes):
        img = np.zeros((H, W, C), dtype=np.float32)
        r0 = (cls * 3) % (H - 6)
        c0 = (cls * 5) % (W - 6)
        color = np.zeros((1, 1, C), dtype=np.float32)
        color[..., cls % C] = 1.0  # cycle RGB channels
        img[r0 : r0 + 6, c0 : c0 + 6, :] = color
        base = img.reshape(-1)
        for _ in range(samples_per_class):
            noise = rng.normal(0.0, 0.10, size=flat_dim).astype(np.float32)
            sample = np.clip(base + noise, 0.0, 1.0)
            images.append(sample.reshape(H, W, C))
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


@register_dataset("cifar10")
def build_cifar10(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    one_hot: bool = True,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for CIFAR‑10.

    In online mode, torchvision can be used if present; otherwise, a
    RuntimeError is raised with an installation hint.
    """

    resolve_cache_dir(cache_dir)
    if offline:
        inputs_raw, labels_raw = _offline_dataset()
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:  # pragma: no cover - optional dependency path
        try:
            import torchvision  # type: ignore
            import torchvision.transforms as T  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "torch/torchvision required for CIFAR-10 online mode. "
                "Install extras: pip install -r requirements-extras.txt"
            ) from exc
        transform = T.Compose([T.ToTensor()])
        train = torchvision.datasets.CIFAR10(str(cache_dir or ".cache"), train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(str(cache_dir or ".cache"), train=False, download=True, transform=transform)
        X = np.concatenate([np.stack([np.array(img) for img, _ in train]), np.stack([np.array(img) for img, _ in test])], axis=0)
        y = np.concatenate([np.array([int(lbl) for _, lbl in train]), np.array([int(lbl) for _, lbl in test])], axis=0)
        inputs_raw, labels_raw = X.astype(np.float32), y.astype(np.int64)
        provenance = {"mode": "download", "source": "torchvision", "n_samples": int(inputs_raw.shape[0])}

    inputs = _prepare_inputs(inputs_raw)
    num_classes = 10
    targets = _prepare_targets(labels_raw, one_hot=one_hot, num_classes=num_classes)

    splits = deterministic_split(inputs.shape[0], val_split=val_split, test_split=test_split, seed=seed)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(inputs, targets, indices, batch_size=batch_size, seed=split_seed)

    data_spec = DataSpec(
        d_in=int(inputs.shape[1]),
        d_out=(num_classes if one_hot else 1),
        task_type="multiclass",
        num_classes=num_classes,
        normalization={"inputs": {"method": "minmax", "range": [0.0, 1.0]}},
        extra={"input_shape": (32, 32, 3)},
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
        name="cifar10",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in split_sizes.items()},
    )


__all__ = ["build_cifar10"]

