from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


def build_optimizer(model: nn.Module, optimizer_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    name = str(optimizer_cfg.get("name", "adamw")).lower()
    betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    eps = float(optimizer_cfg.get("eps", 1.0e-8))
    param_groups = build_param_groups(model, optimizer_cfg)

    if not param_groups:
        raise RuntimeError("No parameters were found for optimizer construction.")

    if name == "adamw":
        return torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(param_groups, betas=betas, eps=eps)
    if name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        nesterov = bool(optimizer_cfg.get("nesterov", False))
        return torch.optim.SGD(param_groups, momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unsupported optimizer name: {name}")


def build_param_groups(model: nn.Module, optimizer_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    named_parameters = list(model.named_parameters())
    if not named_parameters:
        return []

    normalized_group_cfgs = _parse_param_group_cfgs(optimizer_cfg)
    assigned_param_ids: set[int] = set()
    param_groups: list[dict[str, Any]] = []
    default_group_cfg = normalized_group_cfgs.get("default")

    for group_name, group_cfg in normalized_group_cfgs.items():
        if group_name == "default":
            continue
        selected_params = _select_parameters(named_parameters, group_cfg.get("selector"))
        assigned_param_ids.update(id(parameter) for parameter in selected_params)
        if selected_params:
            param_groups.append(_build_optimizer_group(group_name, selected_params, group_cfg))

    if default_group_cfg is not None:
        default_params = [
            parameter
            for _, parameter in named_parameters
            if id(parameter) not in assigned_param_ids
        ]
        if default_params:
            param_groups.append(_build_optimizer_group("default", default_params, default_group_cfg))

    validate_param_groups(model, param_groups)
    return param_groups


def validate_param_groups(model: nn.Module, param_groups: Sequence[Mapping[str, Any]]) -> None:
    named_parameters = list(model.named_parameters())
    model_parameter_map = {id(parameter): name for name, parameter in named_parameters}
    group_membership: dict[int, list[str]] = {}
    unknown_parameters: list[str] = []

    for index, group in enumerate(param_groups):
        group_name = str(group.get("name", f"group_{index}"))
        for parameter in group.get("params", []):
            parameter_name = model_parameter_map.get(id(parameter))
            if parameter_name is None:
                unknown_parameters.append(group_name)
                continue
            group_membership.setdefault(id(parameter), []).append(group_name)

    duplicate_assignments = {
        model_parameter_map[parameter_id]: groups
        for parameter_id, groups in group_membership.items()
        if len(groups) > 1
    }
    missing_parameters = [
        name
        for name, parameter in named_parameters
        if id(parameter) not in group_membership
    ]

    errors: list[str] = []
    if unknown_parameters:
        errors.append(f"optimizer groups include parameters not owned by the model: {unknown_parameters}")
    if duplicate_assignments:
        errors.append(f"parameters assigned to multiple groups: {duplicate_assignments}")
    if missing_parameters:
        errors.append(f"parameters not assigned to any optimizer group: {missing_parameters}")
    if errors:
        raise ValueError("Invalid optimizer param groups. " + " ".join(errors))


def _parse_param_group_cfgs(optimizer_cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    configured_groups = optimizer_cfg.get("param_groups")
    
    if not isinstance(configured_groups, Mapping) or not configured_groups:
        raise ValueError(
            "The optimizer config must contain a valid 'param_groups' mapping. "
            "Legacy flat configurations (e.g., top-level 'backbone_lr') are no longer supported."
        )

    return {
        str(group_name): dict(group_cfg)
        for group_name, group_cfg in configured_groups.items()
    }


def _select_parameters(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    selector_cfg: Mapping[str, Any] | None,
) -> list[nn.Parameter]:
    if selector_cfg is None:
        return [parameter for _, parameter in named_parameters]

    prefixes = tuple(str(prefix) for prefix in selector_cfg.get("prefixes", ()))
    if not prefixes:
        raise ValueError(f"Unsupported optimizer selector config: {dict(selector_cfg)}")

    return [
        parameter
        for name, parameter in named_parameters
        if name.startswith(prefixes)
    ]


def _build_optimizer_group(
    group_name: str,
    parameters: Sequence[nn.Parameter],
    group_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    group: dict[str, Any] = {
        "params": list(parameters),
        "name": group_name,
    }
    for key, value in group_cfg.items():
        if key == "selector":
            continue
        group[key] = value
    return group
