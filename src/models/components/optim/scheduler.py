from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Mapping[str, Any] | None,
    total_steps: int,
) -> dict[str, Any] | None:
    if scheduler_cfg is None:
        return None

    scheduler_cfg = set_default_scheduler(scheduler_cfg)
    name = str(scheduler_cfg.get("name", "linear_warmup_cosine_decay")).lower()
    if name in {"none", "null"}:
        return None
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    interval = str(scheduler_cfg.get("interval", "step"))
    frequency = int(scheduler_cfg.get("frequency", 1))
    group_cfgs = scheduler_cfg["param_groups"]

    lr_lambdas = []
    for index, group in enumerate(optimizer.param_groups):
        group_name = str(group.get("name", f"group_{index}"))
        group_scheduler_cfg = dict(group_cfgs.get(group_name, group_cfgs["default"]))
        lr_lambdas.append(
            _build_scheduler_lambda(
                scheduler_cfg=group_scheduler_cfg,
                total_steps=total_steps,
            )
        )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambdas,
    )
    return {
        "scheduler": scheduler,
        "interval": interval,
        "frequency": frequency,
    }


def set_default_scheduler(scheduler_cfg: Mapping[str, Any]) -> dict[str, Any]:
    normalized_cfg = dict(scheduler_cfg)
    explicit_group_cfgs = normalized_cfg.get("param_groups")
    group_cfgs = (
        {str(group_name): dict(group_cfg) for group_name, group_cfg in explicit_group_cfgs.items()}
        if isinstance(explicit_group_cfgs, Mapping)
        else {}
    )
    base_group_cfg = {
        "name": normalized_cfg.get("name", "linear_warmup_cosine_decay"),
        "warmup_steps": normalized_cfg.get("warmup_steps", 0),
        "min_lr_ratio": normalized_cfg.get("min_lr_ratio", 0.0),
    }
    default_group_cfg = dict(base_group_cfg)
    default_group_cfg.update(group_cfgs.get("default", {}))
    group_cfgs["default"] = default_group_cfg
    normalized_cfg["param_groups"] = group_cfgs
    return normalized_cfg


def _build_scheduler_lambda(
    scheduler_cfg: Mapping[str, Any],
    total_steps: int,
):
    name = str(scheduler_cfg.get("name", "linear_warmup_cosine_decay")).lower()
    if name in {"none", "null"}:
        return lambda current_step: 1.0
    if name == "cosine_decay":
        return _build_cosine_lambda(
            total_steps=total_steps,
            min_lr_ratio=float(scheduler_cfg.get("min_lr_ratio", 0.0)),
        )
    if name == "linear_warmup_cosine_decay":
        return _build_linear_warmup_cosine_lambda(
            total_steps=total_steps,
            warmup_steps=int(scheduler_cfg.get("warmup_steps", 0)),
            min_lr_ratio=float(scheduler_cfg.get("min_lr_ratio", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler name: {name}")


def _build_cosine_lambda(
    total_steps: int,
    min_lr_ratio: float,
):
    bounded_total_steps = max(1, total_steps)

    def lr_lambda(current_step: int) -> float:
        if bounded_total_steps <= 1:
            return 1.0

        progress = current_step / float(bounded_total_steps - 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_lambda


def _build_linear_warmup_cosine_lambda(
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    bounded_warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    cosine_lambda = _build_cosine_lambda(
        total_steps=max(1, total_steps - bounded_warmup_steps),
        min_lr_ratio=min_lr_ratio,
    )

    def lr_lambda(current_step: int) -> float:
        if bounded_warmup_steps > 0 and current_step < bounded_warmup_steps:
            return float(current_step + 1) / float(bounded_warmup_steps)

        if total_steps <= bounded_warmup_steps + 1:
            return 1.0

        return cosine_lambda(current_step - bounded_warmup_steps)

    return lr_lambda
