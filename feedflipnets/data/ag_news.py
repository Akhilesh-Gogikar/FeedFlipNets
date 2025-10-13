"""AG News text dataset via lightweight CSV download + TF-IDF pipeline.

This loader avoids heavyweight NLP dependencies by fetching the widely
used CSV splits and applying scikit-learn TF-IDF followed by Truncated
SVD to a compact dense representation suitable for MLPs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.types import Batch
from .cache import CacheError, fetch
from .registry import DatasetSpec, DataSpec, register_dataset
from .utils import batch_iterator, deterministic_split, resolve_cache_dir

_TRAIN_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
)
_TEST_URL = (
    "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
)


def _offline_dataset(n_features: int, num_classes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic synthetic bag-of-words style dataset."""

    rng = np.random.default_rng(314159)
    samples_per_class = 50
    tokens_per_doc = max(6, n_features // 64)

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
            mask = rng.random(size=n_features) < 0.20
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


def _load_csv(path: Path) -> pd.DataFrame:
    # CSV format: label,title,description (quotes present); labels are 1..4
    return pd.read_csv(path, header=None, names=["label", "title", "text"])


@register_dataset("ag_news")
def build_ag_news(
    *,
    offline: bool = True,
    cache_dir: str | Path | None = None,
    val_split: float = 0.1,
    test_split: float = 0.2,
    seed: int = 0,
    tfidf_max_features: int = 30000,
    svd_dim: int | None = 2048,
) -> DatasetSpec:
    """Create a :class:`DatasetSpec` for AG News (4-class text classification).

    Text is vectorised with TF-IDF (capped vocabulary) and optionally
    projected via Truncated SVD to ``svd_dim`` dense dimensions.
    """

    cache_root = resolve_cache_dir(cache_dir)
    if offline:
        bow_dim = min(max(2048, svd_dim or 2048), tfidf_max_features)
        X, y_idx = _offline_dataset(bow_dim, num_classes=4)
        provenance: dict[str, object] = {"mode": "offline", "source": "synthetic"}
    else:
        try:
            train_path, p_train = fetch(
                name="ag_news_train",
                url=_TRAIN_URL,
                offline_path=cache_root / "ag_news_train.csv",
                cache_dir=cache_root,
                offline=False,
            )
            test_path, p_test = fetch(
                name="ag_news_test",
                url=_TEST_URL,
                offline_path=cache_root / "ag_news_test.csv",
                cache_dir=cache_root,
                offline=False,
            )
            provenance = {"mode": "download", "train": p_train, "test": p_test}
            df_train = _load_csv(train_path)
            df_test = _load_csv(test_path)
            df = pd.concat([df_train, df_test], axis=0, ignore_index=True)
            texts = (df["title"].astype(str) + " " + df["text"].astype(str)).tolist()
            labels = df["label"].astype(int).to_numpy() - 1  # to 0..3
            vec = TfidfVectorizer(
                max_features=int(tfidf_max_features),
                ngram_range=(1, 2),
                lowercase=True,
                dtype=np.float32,
            )
            X_sparse = vec.fit_transform(texts)
            X = X_sparse
            if svd_dim is not None and svd_dim > 0:
                svd = TruncatedSVD(n_components=int(svd_dim), random_state=seed)
                X = svd.fit_transform(X_sparse).astype(np.float32)
            else:
                X = X_sparse.toarray().astype(np.float32)
            y_idx = labels.astype(np.int64)
        except CacheError:
            # Fallback deterministically if download fails
            bow_dim = min(max(2048, svd_dim or 2048), tfidf_max_features)
            X, y_idx = _offline_dataset(bow_dim, num_classes=4)
            provenance = {
                "mode": "offline-fallback",
                "note": "AG News CSV download failed; using synthetic fixture",
            }

    num_classes = 4
    y_one_hot = np.eye(num_classes, dtype=np.float32)[y_idx]

    splits = deterministic_split(X.shape[0], val_split=val_split, test_split=test_split, seed=seed)

    def loader(split: str, batch_size: int) -> Iterator[Batch]:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        indices = getattr(splits, split)
        split_seed = seed + {"train": 0, "val": 1, "test": 2}[split]
        return batch_iterator(X, y_one_hot, indices, batch_size=batch_size, seed=split_seed)

    d_in = int(X.shape[1])
    data_spec = DataSpec(
        d_in=d_in,
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
            "tfidf_max_features": int(tfidf_max_features),
            "svd_dim": int(svd_dim or 0),
        }
    )

    return DatasetSpec(
        name="ag_news",
        loader=loader,
        data_spec=data_spec,
        provenance=provenance,
        splits={k: int(v) for k, v in splits.sizes.items()},
    )


__all__ = ["build_ag_news"]
