# feedflipnets/data/char_lm.py
"""Char-level corpora + batching for the M2b bits-per-char gate.

CFG corpus is the controlled primary: a nested probabilistic grammar whose long-range brackets and
agreement a bigram cannot capture, so the bigram floor lands high (~2.9 bpc) and a real LM (BP) gets
well below it. English (repo docs/*.md prose) is an optional secondary corpus with genuine
irreducible entropy. Float/int only, no torch.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np

Array = np.ndarray


def build_cfg_corpus(seed: int = 0) -> str:
    """Nested probabilistic CFG + larger vocab + long-range brackets/agreement.

    Bigram floor lands high; a real LM (BP) should get well below it.
    """
    rng = np.random.default_rng(seed)
    det = ["the", "a", "this", "that", "my", "his", "her", "some", "no", "every"]
    adj = [
        "quick",
        "brown",
        "lazy",
        "bright",
        "silent",
        "ancient",
        "clever",
        "weary",
        "golden",
        "hollow",
        "restless",
        "crimson",
        "distant",
        "gentle",
    ]
    noun = [
        "fox",
        "dog",
        "river",
        "mountain",
        "scholar",
        "engine",
        "shadow",
        "garden",
        "letter",
        "machine",
        "planet",
        "harbor",
        "melody",
        "cipher",
        "lantern",
        "vulture",
    ]
    verb = [
        "jumps",
        "observes",
        "questions",
        "remembers",
        "constructs",
        "abandons",
        "measures",
        "follows",
        "ignites",
        "dissolves",
        "gathers",
        "translates",
    ]
    adv = ["quickly", "silently", "eventually", "carefully", "rarely", "abruptly", "gladly"]
    conj = ["and", "but", "because", "although", "while", "so"]

    def np_() -> str:
        return f"{rng.choice(det)} {rng.choice(adj)} {rng.choice(noun)}"

    def clause(depth: int = 0) -> str:
        s = f"{np_()} {rng.choice(verb)} {np_()}"
        if rng.random() < 0.35:
            s += f" {rng.choice(adv)}"
        if depth < 2 and rng.random() < 0.4:
            s += f" ({clause(depth + 1)})"
        if depth == 0 and rng.random() < 0.5:
            s += f" {rng.choice(conj)} {clause(depth + 1)}"
        return s

    parts = [clause().capitalize() + "." for _ in range(2200)]
    text = " ".join(parts)
    text = text.replace(". ", ".\n", len(text) // 40)
    return text


def build_english_corpus(root: str = "docs") -> str:
    """Optional secondary corpus: repo docs/*.md prose (genuine irreducible entropy).

    Strips code fences / inline code / URLs lightly; restricts to a stable printable-ASCII vocab.
    """
    import glob
    import re

    chunks = []
    for fp in sorted(glob.glob(root + "/**/*.md", recursive=True)):
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                chunks.append(f.read())
        except OSError:
            pass
    text = "\n".join(chunks)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # drop code blocks
    text = re.sub(r"`[^`]*`", " ", text)  # inline code
    text = re.sub(r"https?://\S+", " ", text)  # urls
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "".join(ch for ch in text if 32 <= ord(ch) < 127 or ch == "\n")
    return text


def prep(text: str) -> Tuple[Array, Array, int]:
    """Char→id, 90/10 train/val split. Returns (train, val, vocab_size)."""
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    split = int(0.9 * len(data))
    return data[:split], data[split:], V


def bigram_floor(train: Array, val: Array, V: int) -> float:
    """Add-1 smoothed bigram val bits-per-char (the 'no structure learned' floor)."""
    bg = defaultdict(Counter)
    for a, b in zip(train[:-1], train[1:]):
        bg[a][b] += 1
    ll = 0.0
    for a, b in zip(val[:-1], val[1:]):
        c = bg[a]
        ll += -math.log2((c[b] + 1) / (sum(c.values()) + V))
    return ll / (len(val) - 1)


def get_batch(src: Array, bs: int, T: int, rng: np.random.Generator) -> Tuple[Array, Array]:
    """Random next-char batch. xb:(bs,T), yb:(bs,T) is xb shifted by one."""
    ix = rng.integers(0, len(src) - T - 1, size=bs)
    xb = np.stack([src[i : i + T] for i in ix])
    yb = np.stack([src[i + 1 : i + T + 1] for i in ix])
    return xb, yb
