"""20 Newsgroups dataset with TF-IDF + Truncated SVD pipeline and offline fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Normalizer

from ..core.types import Batch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import SplitIndices, batch_iterator, deterministic_split, resolve_cache_dir


def _offline_dataset(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic sparse bag-of-words style dataset."""

    rng = np.random.default_rng(4242)
    num_classes = 8
    samples_per_class = 30
    tokens_per_doc = max(6, n_features // 48)

    anchors: list[np.ndarray] = []
    for cls in range(num_classes):
        anchor = np.zeros(n_features, dtype=np.float32)
        active = rng.choice(n_features, size=min(tokens_per_doc * 2, n_features), replace=False)
        anchor[active] = rng.uniform(0.6, 1.0, size=active.size).astype(np.float32)
        anchors.append(anchor)

    documents: list[np.ndarray] = []
    labels: list[int] = []
    for cls, anchor in enumerate(anchors):
        for _ in range(samples_per_class):
            doc = anchor.copy()
            mask = rng.random(size=n_features) < 0.15
            doc *= mask.astype(np.float32)
            noise_idx = rng.choice(
                n_features, size=min(tokens_per_doc // 2 + 1, n_features), replace=False
            )
            doc[noise_idx] += rng.uniform(0.0, 0.3, size=noise_idx.size).astype(np.float32)
            documents.append(doc)
            labels.append(cls)

    X = np.stack(documents, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def _stratified_indices(
    labels: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> SplitIndices:
    """Create deterministic stratified train/val/test partitions."""

    y = labels.astype(np.int64, copy=False)
    indices = np.arange(y.size)

    train_val_idx = indices
    test_idx = np.empty(0, dtype=np.int64)
    if test_split > 0:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
        train_val_idx, test_idx = next(splitter.split(indices, y))

    val_idx = np.empty(0, dtype=np.int64)
    if val_split > 0 and train_val_idx.size > 0:
        remaining = 1.0 - test_split
        rel_val = val_split / remaining if remaining > 0 else 0.0
        rel_val = min(max(rel_val, 0.0), 1.0)
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed + 1)
        train_idx_rel, val_idx_rel = next(splitter.split(train_val_idx, y[train_val_idx]))
        train_idx = train_val_idx[train_idx_rel]
        val_idx = train_val_idx[val_idx_rel]
    else:
        train_idx = train_val_idx

    return SplitIndices(
        train=np.sort(train_idx.astype(np.int64)),
        val=np.sort(val_idx.astype(np.int64)),
        test=np.sort(test_idx.astype(np.int64)),
    )


def _build_real_representation(
    *,
    cache_root: Path,
    subset: str,
    seed: int,
    val_split: float,
    test_split: float,
    tfidf_max_features: int | None,
    svd_dim: int | None,
    min_df: int,
    max_df: float,
    ngram_range: tuple[int, int],
    stop_words: str | None,
    sublinear_tf: bool,
) -> tuple[np.ndarray, np.ndarray, SplitIndices, dict]:
    raw = fetch_20newsgroups(
        subset=subset,
        remove=("headers", "footers"),
        data_home=str(cache_root),
    )
    texts = np.asarray(raw.data, dtype=object)
    labels = raw.target.astype(np.int64)
    splits = _stratified_indices(labels, val_split, test_split, seed)

    max_feats = int(tfidf_max_features) if tfidf_max_features else None
    vec = TfidfVectorizer(
        max_features=max_feats,
        min_df=int(min_df),
        max_df=float(max_df),
        ngram_range=tuple(ngram_range),
        stop_words=stop_words,
        lowercase=True,
        norm="l2",
        sublinear_tf=sublinear_tf,
    )

    train_texts = texts[splits.train]
    train_sparse = vec.fit_transform(train_texts)
    full_sparse = vec.transform(texts)

    vocab_size = len(vec.vocabulary_)
    svd_used = None
    normalizer = None

    if svd_dim is not None and svd_dim > 0 and train_sparse.shape[1] > 1:
        n_components = min(int(svd_dim), train_sparse.shape[1] - 1)
        n_components = max(n_components, 1)
        svd_used = TruncatedSVD(n_components=n_components, random_state=seed)
        normalizer = Normalizer(copy=False)
        reduced = svd_used.fit_transform(train_sparse).astype(np.float32)
        reduced = normalizer.fit_transform(reduced)
        dense = svd_used.transform(full_sparse).astype(np.float32)
        dense = normalizer.transform(dense)
    else:
        dense = full_sparse.toarray().astype(np.float32)

    features = np.ascontiguousarray(dense.astype(np.float32))
    provenance = {
        "mode": "download",
        "subset": subset,
        "tfidf_vocabulary": int(vocab_size),
        "tfidf_max_features": int(max_feats or 0),
        "svd_dim": int(svd_dim or 0),
        "min_df": int(min_df),
        "max_df": float(max_df),
        "ngram_range": list(ngram_range),
        "stop_words": stop_words or "none",
        "sublinear_tf": bool(sublinear_tf),
        "target_names": list(raw.target_names),
    }
    if svd_used is not None:
        provenance["svd_explained_variance"] = float(svd_used.explained_variance_ratio_.sum())

    return features, labels, splits, provenance


@register_dataset("20newsgroups")
def build_20newsgroups(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    subset: str = "all",
    n_features: int | None = 4096,
    tfidf_max_features: int | None = 60000,
    svd_dim: int | None = None,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple[int, int] | list[int] | tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    sublinear_tf: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for 20 Newsgroups."""

    if isinstance(ngram_range, (list, tuple)) and len(ngram_range) == 2:
        ngram = (int(ngram_range[0]), int(ngram_range[1]))
    else:  # pragma: no cover - guardrail
        raise ValueError("ngram_range must be a pair such as (1, 2)")

    effective_svd = int(svd_dim or n_features or 2048)

    if offline:
        feature_dim = int(n_features or 4096)
        X, y = _offline_dataset(feature_dim)
        provenance: dict[str, object] = {
            "mode": "offline",
            "source": "synthetic",
            "feature_dim": feature_dim,
        }
        splits = deterministic_split(
            X.shape[0], val_split=val_split, test_split=test_split, seed=seed
        )
    else:
        cache_root = resolve_cache_dir(cache_dir)
        max_feats = tfidf_max_features or max(effective_svd * 4, 50000)
        X, y, splits, provenance = _build_real_representation(
            cache_root=cache_root,
            subset=subset,
            seed=seed,
            val_split=val_split,
            test_split=test_split,
            tfidf_max_features=max_feats,
            svd_dim=effective_svd,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram,
            stop_words=stop_words,
            sublinear_tf=sublinear_tf,
        )

    num_classes = int(np.max(y)) + 1 if y.size else 0
    y_one_hot = np.eye(num_classes, dtype=np.float32)[y] if num_classes else y.reshape(-1, 1)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(
            X,
            y_one_hot,
            indices,
            batch_size=batch_size,
            seed=split_seed,
            replacement=(split == "train"),
        )

    data_spec = DataSpec(
        d_in=int(X.shape[1]),
        d_out=num_classes,
        task_type="multiclass",
        num_classes=num_classes,
        normalization={"inputs": {"method": "l2"}},
    )

    provenance.update(
        {
            "val_split": val_split,
            "test_split": test_split,
            "seed": seed,
            "svd_dim": effective_svd,
        }
    )

    return DatasetSpec(
        name="20newsgroups",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["build_20newsgroups"]
