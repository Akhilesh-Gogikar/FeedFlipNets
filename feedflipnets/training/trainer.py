"""Deterministic, modality-aware training loops for FeedFlipNets."""

from __future__ import annotations

import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableSequence, Sequence

import numpy as np

from ..core.activations import relu
from ..core.quant import quantize_ternary_det, quantize_ternary_stoch
from ..core.strategies import Backprop, FeedbackStrategy
from ..core.types import (
    ActivationState,
    Array,
    Batch,
    Gradients,
    ModelDescription,
    RunResult,
    StrategyState,
)
from .losses import REGISTRY as LOSS_REGISTRY
from .metrics import compute_metrics, default_metrics


def _relu_deriv(z: Array) -> Array:
    return (z > 0).astype(np.float32)


def _ternary_zero_ratio(model: "FeedForwardModel") -> float:
    total = sum(int(w.size) for w in model.weights)
    if total == 0:
        return 0.0
    zeros = 0
    for w in model.weights:
        zeros += int(np.count_nonzero(np.isclose(w, 0.0)))
    return float(zeros / total)


@dataclass
class FeedForwardModel:
    """Lightweight feed-forward network with ternary quantisation."""

    layer_dims: Sequence[int]
    tau: float
    quant: str = "det"
    seed: int = 0
    weights: MutableSequence[Array] = field(init=False, repr=False)
    shadow_weights: MutableSequence[Array] = field(init=False, repr=False)
    last_quant_stats: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset(self.seed)

    def describe(self) -> ModelDescription:
        return ModelDescription(layer_dims=list(self.layer_dims))

    def reset(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._quant_rng = np.random.default_rng(seed + 1)
        shadow: list[Array] = []
        dims = list(self.layer_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            W = rng.standard_normal((in_dim, out_dim), dtype=np.float32) * 0.05
            shadow.append(W)
        self.shadow_weights = shadow
        self.weights = [w.copy() for w in shadow]
        self.last_quant_stats = []

    def forward(self, inputs: Array) -> tuple[Array, ActivationState]:
        layer_inputs: list[Array] = [inputs]
        layer_derivs: list[Array] = []
        x = inputs
        for idx, W in enumerate(self.weights):
            z = x @ W
            if idx < len(self.weights) - 1:
                layer_derivs.append(_relu_deriv(z))
                x = relu(z)
                layer_inputs.append(x)
            else:
                x = z
        activations = ActivationState(
            layer_inputs=layer_inputs,
            layer_derivs=layer_derivs,
            weights=[w.copy() for w in self.weights],
        )
        return x, activations

    def apply_gradients(self, grads: Gradients) -> None:
        for idx, W in enumerate(self.shadow_weights):
            grad = grads.get(f"W{idx}")
            if grad is None:
                continue
            self.shadow_weights[idx] = W + grad

    def copy_shadow_to_forward(self) -> None:
        self.weights = [w.copy() for w in self.shadow_weights]
        self.last_quant_stats = [0.0 for _ in self.weights]

    def quantise(self) -> list[float]:
        stats: list[float] = []
        new_forward: list[Array] = []
        for idx, shadow in enumerate(self.shadow_weights):
            if self.quant == "det":
                quantised = quantize_ternary_det(shadow, self.tau)
            elif self.quant == "stoch":
                quantised = quantize_ternary_stoch(shadow, self.tau, self._quant_rng)
            else:  # pragma: no cover - guardrail
                raise ValueError(f"Unknown quantisation mode: {self.quant}")
            diff = quantised.astype(np.float32) - shadow.astype(np.float32)
            stats.append(float(np.mean(diff * diff)))
            new_forward.append(quantised)
        self.weights = new_forward
        self.last_quant_stats = stats
        return stats

    def state_dict(self) -> Mapping[str, Array]:
        payload: dict[str, Array] = {}
        for idx, shadow in enumerate(self.shadow_weights):
            payload[f"W{idx}"] = shadow.copy()
            payload[f"W{idx}_forward"] = self.weights[idx].copy()
        return payload

    def load_state_dict(self, state: Mapping[str, Array]) -> None:
        shadow: list[Array] = []
        forward: list[Array] = []
        idx = 0
        while True:
            key = f"W{idx}"
            if key not in state:
                if idx == 0:
                    raise KeyError(f"Missing weight {key} in state dict")
                break
            shadow_w = state[key].copy()
            shadow.append(shadow_w)
            fwd_key = f"W{idx}_forward"
            forward.append(state.get(fwd_key, shadow_w).copy())
            idx += 1
        self.shadow_weights = shadow
        self.weights = forward
        self.last_quant_stats = [0.0 for _ in forward]

    def parameter_count(self) -> int:
        return int(sum(int(w.size) for w in self.shadow_weights))


@dataclass
class SGDOptimizer:
    """Vanilla SGD with optional momentum (unused but extendable)."""

    lr: float

    def step(self, model: FeedForwardModel, grads: Gradients) -> None:
        scaled = {name: -self.lr * grad for name, grad in grads.items()}
        model.apply_gradients(scaled)


class Trainer:
    """Run deterministic training loops with pluggable feedback strategies."""

    def __init__(
        self,
        model: FeedForwardModel,
        strategy: FeedbackStrategy,
        optimizer: SGDOptimizer,
        callbacks: Sequence[object] | None = None,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.optimizer = optimizer
        self.callbacks = list(callbacks or [])
        self._state: StrategyState | None = None
        self._timings: Dict[str, list[float]] = {"train": [], "val": [], "test": []}

    def run(
        self,
        dataloader: Iterable[Batch] | Mapping[str, Iterable[Batch]],
        epochs: int,
        seed: int,
        device: str = "cpu",
        *,
        determinism: bool = True,
        steps_per_epoch: int | None = None,
        val_loader: Iterable[Batch] | None = None,
        test_loader: Iterable[Batch] | None = None,
        val_steps: int | None = None,
        test_steps: int | None = None,
        task_type: str = "regression",
        num_classes: int | None = None,
        loss: str = "auto",
        metric_names: Sequence[str] | str = (),
        eval_every: int = 1,
        split_loggers: Mapping[str, Sequence[object]] | None = None,
        flip: str = "ternary",
        flip_schedule: str | None = None,
        ternary_mode: str | None = None,
        early_stopping_patience: int | None = None,
        checkpoint_dir: str | Path | None = None,
        grad_clip: float | None = None,
        alignment_probe_steps: int | None = None,
        eval_on_shadow: bool = False,
    ) -> RunResult:
        if device != "cpu":  # pragma: no cover - guardrail
            raise ValueError("Only CPU execution is supported in the reference trainer")

        if isinstance(dataloader, Mapping):
            train_loader = dataloader.get("train")
            if train_loader is None:
                raise ValueError("train loader missing from dataloader mapping")
            val_loader = val_loader or dataloader.get("val")
            test_loader = test_loader or dataloader.get("test")
        else:
            train_loader = dataloader

        if isinstance(metric_names, str):
            if metric_names == "default" or metric_names.strip() == "":
                metric_names = default_metrics(task_type, num_classes=num_classes)
            else:
                metric_names = [m.strip() for m in metric_names.split(",") if m.strip()]
        if not metric_names:
            metric_names = default_metrics(task_type, num_classes=num_classes)

        loss_fn = LOSS_REGISTRY.resolve(loss, task_type=task_type)
        self._set_seed(seed, determinism)
        self.model.reset(seed)
        state = self.strategy.init(self.model.describe())
        train_steps = steps_per_epoch or self._infer_steps(train_loader)
        if ternary_mode is not None:
            warnings.warn(
                "`ternary_mode` is deprecated; use `flip_schedule` instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            if flip_schedule is not None:
                raise ValueError("Cannot specify both `flip_schedule` and `ternary_mode`.")
            flip_schedule = ternary_mode

        flip = (flip or "ternary").lower()
        if flip not in {"off", "ternary"}:
            raise ValueError("flip must be one of {'off','ternary'}")

        if flip == "off":
            resolved_schedule = "off"
        else:
            resolved_schedule = (flip_schedule or "per_step").lower()

        if resolved_schedule not in {"off", "per_step", "per_epoch"}:
            raise ValueError("flip_schedule must be one of {'off','per_step','per_epoch'}")

        flip_enabled = flip != "off"

        best_loss = float("inf")
        best_state: Mapping[str, Array] | None = None
        epochs_no_improve = 0
        total_steps = 0
        split_loggers = split_loggers or {}
        checkpoint_dir = Path(checkpoint_dir or ".")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            train_metrics, state = self._run_phase(
                train_loader,
                train_steps,
                state,
                loss_fn,
                metric_names,
                task_type,
                num_classes,
                training=True,
                flip_schedule=resolved_schedule,
                flip_enabled=flip_enabled,
                split_name="train",
                grad_clip=grad_clip,
                alignment_probe_steps=alignment_probe_steps,
            )
            total_steps += train_steps
            self._emit_epoch("train", epoch, train_metrics, split_loggers)

            should_eval = epoch % max(1, eval_every) == 0
            val_metrics = None
            if should_eval and val_loader is not None and (val_steps or 0) > 0:
                val_metrics, _ = self._run_phase(
                    val_loader,
                    val_steps or self._infer_steps(val_loader),
                    state,
                    loss_fn,
                    metric_names,
                    task_type,
                    num_classes,
                    training=False,
                    flip_schedule="off",
                    flip_enabled=False,
                    split_name="val",
                    grad_clip=None,
                    eval_quantized=flip_enabled and not eval_on_shadow,
                )
                self._emit_epoch("val", epoch, val_metrics, split_loggers)

            if should_eval and test_loader is not None and (test_steps or 0) > 0:
                test_metrics, _ = self._run_phase(
                    test_loader,
                    test_steps or self._infer_steps(test_loader),
                    state,
                    loss_fn,
                    metric_names,
                    task_type,
                    num_classes,
                    training=False,
                    flip_schedule="off",
                    flip_enabled=False,
                    split_name="test",
                    grad_clip=None,
                    eval_quantized=flip_enabled and not eval_on_shadow,
                )
                self._emit_epoch("test", epoch, test_metrics, split_loggers)

            target_metrics = val_metrics or train_metrics
            current_loss = float(target_metrics.get("loss", 0.0))
            if current_loss < best_loss - 1e-9:
                best_loss = current_loss
                epochs_no_improve = 0
                best_state = self.model.state_dict()
                self._save_checkpoint(checkpoint_dir / "best.ckpt", best_state)
            else:
                epochs_no_improve += 1
                if early_stopping_patience and epochs_no_improve >= early_stopping_patience:
                    break

        last_state = self.model.state_dict()
        self._save_checkpoint(checkpoint_dir / "last.ckpt", last_state)
        self._state = state
        return RunResult(
            steps=total_steps,
            metrics_path="",
            manifest_path="",
            summary_path="",
        )

    def timings(self) -> Mapping[str, Sequence[float]]:
        return {split: list(values) for split, values in self._timings.items() if values}

    # ------------------------------------------------------------------
    # Internal helpers

    def _run_phase(
        self,
        loader: Iterable[Batch],
        steps: int,
        state: StrategyState,
        loss_fn,
        metric_names: Sequence[str],
        task_type: str,
        num_classes: int | None,
        *,
        training: bool,
        flip_schedule: str,
        flip_enabled: bool,
        split_name: str,
        grad_clip: float | None,
        alignment_probe_steps: int | None = None,
        eval_quantized: bool = False,
    ) -> tuple[Mapping[str, float], StrategyState]:
        iterator = iter(loader)
        losses: list[float] = []
        preds_all: list[Array] = []
        targets_all: list[Array] = []
        current_state = state
        total_samples = 0
        start = time.perf_counter()
        # Alignment accumulators (per-layer) when requested
        want_align = bool(training and (alignment_probe_steps or 0) > 0)
        align_bp = Backprop() if want_align else None
        align_sums: list[float] = []
        align_counts: list[int] = []
        sign_match_counts: list[int] = []
        sign_match_totals: list[int] = []
        quant_var_sums: list[float] = []
        quant_var_steps = 0
        grad_norm_sum = 0.0
        grad_norm_count = 0

        saved_forward: list[Array] | None = None
        saved_quant_stats: list[float] | None = None
        if not training:
            if eval_quantized:
                # Evaluate the DEPLOYED (ternary-projected) weights, not the raw
                # float latents. Use the deterministic projection so eval never
                # advances the stochastic quantisation RNG, and restore the
                # forward weights afterwards so training state is untouched.
                saved_forward = [w.copy() for w in self.model.weights]
                saved_quant_stats = list(self.model.last_quant_stats)
                self.model.weights = [
                    quantize_ternary_det(shadow, self.model.tau)
                    for shadow in self.model.shadow_weights
                ]
            else:
                self.model.copy_shadow_to_forward()
        elif not flip_enabled:
            self.model.copy_shadow_to_forward()

        for step_idx in range(max(1, steps)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            predictions, activations = self.model.forward(batch.inputs)
            loss_value, delta = loss_fn(predictions, batch.targets)
            losses.append(loss_value)
            preds_all.append(predictions.copy())
            targets_all.append(batch.targets.copy())
            total_samples += int(batch.inputs.shape[0])
            if training:
                grads, current_state = self.strategy.backward(activations, delta, current_state)
                # Alignment probe on early steps
                if want_align and step_idx < int(alignment_probe_steps or 0):
                    bp_grads, _ = align_bp.backward(
                        activations, delta, StrategyState()
                    )  # type: ignore[arg-type]
                    # Initialize accumulators lazily
                    if not align_sums:
                        last_idx = len(activations.weights) - 1
                        align_sums = [0.0 for _ in range(last_idx + 1)]
                        align_counts = [0 for _ in range(last_idx + 1)]
                        sign_match_counts = [0 for _ in range(last_idx + 1)]
                        sign_match_totals = [0 for _ in range(last_idx + 1)]
                    for name, g_dfa in grads.items():
                        if name in bp_grads:
                            g_bp = bp_grads[name]
                            a = float(np.sum(g_dfa.astype(np.float64) * g_bp.astype(np.float64)))
                            b = float(np.sqrt(np.sum(g_dfa.astype(np.float64) ** 2)) + 1e-12)
                            c = float(np.sqrt(np.sum(g_bp.astype(np.float64) ** 2)) + 1e-12)
                            rho = a / (b * c)
                            idx = int(name[1:]) if name.startswith("W") else 0
                            if 0 <= idx < len(align_sums):
                                align_sums[idx] += rho
                                align_counts[idx] += 1
                                mask = np.logical_or(
                                    np.abs(g_bp) > 0.0,
                                    np.abs(g_dfa) > 0.0,
                                )
                                total = int(np.count_nonzero(mask))
                                if total > 0:
                                    matches = int(
                                        np.count_nonzero(
                                            np.sign(g_bp[mask]) == np.sign(g_dfa[mask])
                                        )
                                    )
                                    sign_match_counts[idx] += matches
                                    sign_match_totals[idx] += total
                if grad_clip is not None and grad_clip > 0:
                    # Compute global L2 norm across all parameter gradients
                    sqsum = 0.0
                    for g in grads.values():
                        sqsum += float(np.sum(g.astype(np.float64) ** 2))
                    norm = float(np.sqrt(sqsum))
                    if norm > grad_clip:
                        scale = float(grad_clip / (norm + 1e-12))
                        for k, g in grads.items():
                            grads[k] = g * scale
                grad_sq = 0.0
                for g in grads.values():
                    grad_sq += float(np.sum(g.astype(np.float64) ** 2))
                grad_norm_sum += float(np.sqrt(grad_sq))
                grad_norm_count += 1
                self.optimizer.step(self.model, grads)
                if flip_enabled:
                    if flip_schedule == "per_step":
                        variances = self.model.quantise()
                        if variances:
                            if not quant_var_sums:
                                quant_var_sums = [0.0 for _ in variances]
                            for idx, val in enumerate(variances):
                                quant_var_sums[idx] += float(val)
                            quant_var_steps += 1
                    elif flip_schedule == "off":
                        self.model.copy_shadow_to_forward()
                else:
                    self.model.copy_shadow_to_forward()

        if training and flip_enabled and flip_schedule == "per_epoch":
            variances = self.model.quantise()
            if variances:
                if not quant_var_sums:
                    quant_var_sums = [0.0 for _ in variances]
                for idx, val in enumerate(variances):
                    quant_var_sums[idx] += float(val)
                quant_var_steps += 1

        metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
        if preds_all:
            predictions = np.concatenate(preds_all, axis=0)
            targets = np.concatenate(targets_all, axis=0)
            metrics.update(
                compute_metrics(
                    metric_names,
                    predictions,
                    targets,
                    task_type=task_type,
                    num_classes=num_classes,
                )
            )

        elapsed = max(time.perf_counter() - start, 1e-9)
        self._timings.setdefault(split_name, []).append(elapsed)
        metrics["sample_count"] = float(total_samples)
        metrics["samples_per_step"] = float(total_samples / max(steps, 1))
        metrics["ternary_zero_ratio"] = _ternary_zero_ratio(self.model)
        # Emit average alignment across probed steps if collected
        if want_align and align_sums and any(align_counts):
            rhos: list[float] = []
            deficits: list[float] = []
            for i, (s, n) in enumerate(zip(align_sums, align_counts)):
                if n > 0:
                    avg = float(s / n)
                    metrics[f"rho_l{i}"] = avg
                    rhos.append(avg)
                    deficit = float(1.0 - avg)
                    metrics[f"rho_deficit_l{i}"] = deficit
                    deficits.append(deficit)
            if rhos:
                metrics["rho_mean"] = float(np.mean(rhos))
                metrics["rho_min"] = float(np.min(rhos))
            if deficits:
                metrics["rho_deficit_mean"] = float(np.mean(deficits))
                metrics["rho_deficit_max"] = float(np.max(deficits))
        if want_align and sign_match_totals:
            probs: list[float] = []
            for i, total in enumerate(sign_match_totals):
                if total > 0:
                    prob = float(sign_match_counts[i] / total)
                    metrics[f"p_align_l{i}"] = prob
                    probs.append(prob)
            if probs:
                metrics["p_align_mean"] = float(np.mean(probs))
                metrics["p_align_min"] = float(np.min(probs))
        if grad_norm_count > 0:
            metrics["grad_norm_mean"] = float(grad_norm_sum / grad_norm_count)
        if quant_var_steps > 0 and quant_var_sums:
            per_layer = [float(val / quant_var_steps) for val in quant_var_sums]
            metrics["sigma_q2_mean"] = float(np.mean(per_layer))
            metrics["sigma_q2_max"] = float(np.max(per_layer))
            for i, val in enumerate(per_layer):
                metrics[f"sigma_q2_l{i}"] = float(val)
        if saved_forward is not None:
            self.model.weights = saved_forward
            self.model.last_quant_stats = saved_quant_stats or [0.0 for _ in saved_forward]
        return metrics, current_state

    def _emit_epoch(
        self,
        split: str,
        epoch: int,
        metrics: Mapping[str, float],
        loggers: Mapping[str, Sequence[object]],
    ) -> None:
        for callback in self.callbacks:
            if hasattr(callback, "on_epoch"):
                callback.on_epoch(epoch, metrics)  # type: ignore[attr-defined]
        for callback in loggers.get(split, []):
            if hasattr(callback, "on_epoch"):
                callback.on_epoch(epoch, metrics)  # type: ignore[attr-defined]
            elif callable(callback):
                callback(epoch, metrics)

    @staticmethod
    def _set_seed(seed: int, determinism: bool) -> None:
        if determinism:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _infer_steps(dataloader: Iterable[Batch]) -> int:
        if hasattr(dataloader, "__len__"):
            try:
                return len(dataloader)  # type: ignore[arg-type]
            except TypeError:  # pragma: no cover - defensive
                pass
        return 1

    @staticmethod
    def _save_checkpoint(path: Path, state: Mapping[str, Array]) -> None:
        payload = {name: value for name, value in state.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(handle, **payload)


__all__ = ["FeedForwardModel", "SGDOptimizer", "Trainer"]
