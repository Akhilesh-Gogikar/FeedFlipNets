#!/usr/bin/env python3
"""Minimal CNN baselines: BP vs DFA on Fashion‑MNIST/CIFAR‑10.

This script is intentionally self‑contained and optional. It uses PyTorch
to implement a tiny CNN and a simple DFA updater for conv/linear layers.

Usage examples (CPU):

  # Backprop on Fashion‑MNIST
  python scripts/baselines/cnn_dfa_baselines.py --dataset fashion_mnist --method bp \
      --epochs 2 --batch-size 128 --run-dir runs/baselines/fmnist-bp

  # DFA on Fashion‑MNIST (expect significant degradation vs BP)
  python scripts/baselines/cnn_dfa_baselines.py --dataset fashion_mnist --method dfa \
      --epochs 2 --batch-size 128 --run-dir runs/baselines/fmnist-dfa

  # Backprop on CIFAR‑10 (slightly heavier)
  python scripts/baselines/cnn_dfa_baselines.py --dataset cifar10 --method bp \
      --epochs 2 --batch-size 128 --run-dir runs/baselines/cifar10-bp

Installing extras:
  pip install torch torchvision
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


class TinyCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        # Keep padding=1 to preserve spatial dims across conv layers
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2)
        # For 28x28 -> 14x14 -> 7x7; for 32x32 -> 16x16 -> 8x8
        self.head: nn.Linear | None = None
        self.num_classes = num_classes
        self._shape = None  # set on first forward pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        z1 = self.conv1(x)
        a1 = F.relu(z1)
        p1 = self.pool(a1)
        z2 = self.conv2(p1)
        a2 = F.relu(z2)
        p2 = self.pool(a2)
        b, c, h, w = p2.shape
        if self.head is None:
            self.head = nn.Linear(c * h * w, self.num_classes, bias=True)
            # Move head to same device as p2
            self.head.to(p2.device)
        f = p2.reshape(b, -1)
        logits = self.head(f)
        cache = {
            "z1": z1,
            "a1": a1,
            "p1": p1,
            "z2": z2,
            "a2": a2,
            "p2": p2,
            "f": f,
        }
        return logits, cache


class DFAUpdater:
    """Manual DFA updates for TinyCNN using fixed feedback matrices.

    - For conv layers, project the output error to each layer's pre‑activation
      via a fixed matrix and use dReLU gating. Weight gradients are computed via
      conv2d_weight gradients.
    - For the final linear layer, use the standard gradient (delta = e).
    """

    def __init__(self, model: TinyCNN, num_classes: int, seed: int = 0, lr: float = 0.01) -> None:
        self.model = model
        self.num_classes = num_classes
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.lr = lr
        self._feedback = {}

    @staticmethod
    def _relu_deriv(z: torch.Tensor) -> torch.Tensor:
        return (z > 0).to(dtype=z.dtype)

    def _get_feedback(self, key: str, out_dim: int, in_dim: int) -> torch.Tensor:
        if key in self._feedback:
            return self._feedback[key]
        # Fan‑out: classes -> layer preact dim; column‑normalize for stability
        B = torch.randn((self.num_classes, in_dim), generator=self.rng) / (out_dim ** 0.5)
        # Normalize columns to unit norm
        B = B / (B.norm(dim=0, keepdim=True) + 1e-8)
        self._feedback[key] = B.to(dtype=torch.float32)
        return self._feedback[key]

    @torch.no_grad()
    def step(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        model = self.model
        model.train()
        logits, cache = model(x)
        bsz = x.shape[0]

        # Cross‑entropy gradient wrt logits: softmax - one_hot
        probs = F.softmax(logits, dim=1)
        e = probs - F.one_hot(y, num_classes=self.num_classes).to(probs.dtype)
        e = e / float(bsz)

        # Final linear layer gradient (standard)
        f = cache["f"]  # (N, F)
        gW3 = f.t().mm(e)
        gb3 = e.sum(dim=0)

        # Project error to conv2 pre‑activations
        z2 = cache["z2"]
        pre2_dim = int(z2[0].numel())
        B2 = self._get_feedback("conv2", self.num_classes, pre2_dim)
        delta2 = e.mm(B2).reshape_as(z2)
        delta2 *= self._relu_deriv(z2)

        # Gradients for conv2 weights/bias
        # Weight shape: (out_c=32, in_c=16, k=3,3)
        p1 = cache["p1"]
        gW2 = torch.nn.grad.conv2d_weight(
            input=p1, weight_size=model.conv2.weight.shape, grad_output=delta2, stride=1, padding=1
        )
        gb2 = delta2.sum(dim=(0, 2, 3))

        # Project error to conv1 pre‑activations
        z1 = cache["z1"]
        pre1_dim = int(z1[0].numel())
        B1 = self._get_feedback("conv1", self.num_classes, pre1_dim)
        delta1 = e.mm(B1).reshape_as(z1)
        delta1 *= self._relu_deriv(z1)

        # Gradients for conv1 weights/bias
        gW1 = torch.nn.grad.conv2d_weight(
            input=x, weight_size=model.conv1.weight.shape, grad_output=delta1, stride=1, padding=1
        )
        gb1 = delta1.sum(dim=(0, 2, 3))

        # SGD update
        model.conv1.weight.add_(gW1, alpha=-self.lr)
        model.conv1.bias.add_(gb1, alpha=-self.lr)
        model.conv2.weight.add_(gW2, alpha=-self.lr)
        model.conv2.bias.add_(gb2, alpha=-self.lr)
        model.head.weight.add_(gW3.t(), alpha=-self.lr)
        model.head.bias.add_(gb3, alpha=-self.lr)

        # Loss for logging
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        return {"loss": float(loss.item()), "acc": float(acc)}


def get_data(dataset: str, batch_size: int, limit: int | None = None, seed: int = 0):
    torch.manual_seed(seed)
    if dataset == "fashion_mnist":
        transform = T.Compose([T.ToTensor()])
        train = torchvision.datasets.FashionMNIST(".cache", train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(".cache", train=False, download=True, transform=transform)
        in_ch = 1
        num_classes = 10
    elif dataset == "cifar10":
        transform = T.Compose([T.ToTensor()])
        train = torchvision.datasets.CIFAR10(".cache", train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(".cache", train=False, download=True, transform=transform)
        in_ch = 3
        num_classes = 10
    else:
        raise ValueError("dataset must be one of {'fashion_mnist','cifar10'}")

    if limit is not None and limit > 0:
        train = torch.utils.data.Subset(train, range(min(limit, len(train))))
        test = torch.utils.data.Subset(test, range(min(max(limit // 5, 1000), len(test))))

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, in_ch, num_classes


def train_bp(model: TinyCNN, loaders, epochs: int, lr: float, device: str = "cpu") -> dict:
    train_loader, test_loader = loaders
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    metrics = {"train": [], "test": []}
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, acc_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().sum().item()
                loss_sum += float(loss.item()) * x.size(0)
                acc_sum += acc
                n += x.size(0)
        train_rec = {"epoch": epoch, "loss": loss_sum / max(1, n), "acc": acc_sum / max(1, n)}
        metrics["train"].append(train_rec)

        # Eval
        model.eval()
        with torch.no_grad():
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().sum().item()
                loss_sum += float(loss.item()) * x.size(0)
                acc_sum += acc
                n += x.size(0)
        test_rec = {"epoch": epoch, "loss": loss_sum / max(1, n), "acc": acc_sum / max(1, n)}
        metrics["test"].append(test_rec)
        print(json.dumps({"split": "train", **train_rec}))
        print(json.dumps({"split": "test", **test_rec}))
    return metrics


def train_dfa(model: TinyCNN, loaders, epochs: int, lr: float, num_classes: int, device: str = "cpu", seed: int = 0) -> dict:
    train_loader, test_loader = loaders
    model.to(device)
    # Turn off autograd to avoid accumulating gradients
    for p in model.parameters():
        p.requires_grad_(False)

    updater = DFAUpdater(model, num_classes=num_classes, seed=seed, lr=lr)
    metrics = {"train": [], "test": []}
    for epoch in range(1, epochs + 1):
        # Train
        loss_sum, acc_sum, n = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            rec = updater.step(x, y)
            loss_sum += rec["loss"] * x.size(0)
            acc_sum += rec["acc"] * x.size(0)
            n += x.size(0)
        train_rec = {"epoch": epoch, "loss": loss_sum / max(1, n), "acc": acc_sum / max(1, n)}
        metrics["train"].append(train_rec)

        # Eval
        model.eval()
        with torch.no_grad():
            loss_sum, acc_sum, n = 0.0, 0.0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().sum().item()
                loss_sum += float(loss.item()) * x.size(0)
                acc_sum += acc
                n += x.size(0)
        test_rec = {"epoch": epoch, "loss": loss_sum / max(1, n), "acc": acc_sum / max(1, n)}
        metrics["test"].append(test_rec)
        print(json.dumps({"split": "train", **train_rec}))
        print(json.dumps({"split": "test", **test_rec}))
        model.train()
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["fashion_mnist", "cifar10"], default="fashion_mnist")
    p.add_argument("--method", choices=["bp", "dfa"], default="bp")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=5000, help="Limit training samples for quick runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--run-dir", type=Path, default=Path("runs/baselines"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    train_loader, test_loader, in_ch, num_classes = get_data(args.dataset, args.batch_size, args.limit, seed=args.seed)
    model = TinyCNN(in_ch=in_ch, num_classes=num_classes)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    if args.method == "bp":
        metrics = train_bp(model, (train_loader, test_loader), epochs=args.epochs, lr=args.lr, device=args.device)
    else:
        metrics = train_dfa(
            model,
            (train_loader, test_loader),
            epochs=args.epochs,
            lr=args.lr,
            num_classes=num_classes,
            device=args.device,
            seed=args.seed,
        )
    # Persist summary JSON
    out = args.run_dir / f"{args.dataset}_{args.method}_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"saved": str(out)}))


if __name__ == "__main__":
    main()

