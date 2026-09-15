from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import yaml


FORBIDDEN_SOURCE_TOKENS = (
    "wfgen",
    "synthetic dag",
    "streets",
    "gurnee",
    "uci traffic",
    "m/g/1",
)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path, seen: set[Path]) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"cyclic config inheritance: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {resolved}")
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    parent_path = (resolved.parent / str(parent)).resolve()
    return _deep_merge(_read_yaml(parent_path, seen | {resolved}), payload)


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        flattened: list[str] = []
        for key, item in value.items():
            flattened.append(str(key))
            flattened.extend(_flatten_text(item))
        return flattened
    if isinstance(value, (list, tuple, set)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_text(item))
        return flattened
    return [str(value)]


def validate_config(config: Mapping[str, Any]) -> None:
    text = " ".join(_flatten_text(config)).lower()
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise ValueError(f"forbidden data source or model found: {token}")

    required = ("data", "scenario", "timing", "resources", "training", "runtime")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing required configuration sections: {missing}")
    calibration_contract = config.get("calibration")
    if not isinstance(calibration_contract, Mapping):
        raise ValueError("missing calibration contract")
    if calibration_contract.get("update_during_training") is not False:
        raise ValueError("calibration.update_during_training must be false")
    for key in ("manifest", "frozen_bounds"):
        if not isinstance(calibration_contract.get(key), str) or not calibration_contract[key]:
            raise ValueError(f"calibration.{key} must be a non-empty path")

    scenario = config["scenario"]
    if any(int(scenario[name]) <= 0 for name in ("ugv_count", "rsu_count", "uav_count")):
        raise ValueError("node counts must be positive")
    if int(scenario.get("active_workflows_per_episode", 0)) <= 0:
        raise ValueError("active workflows per episode must be positive")
    if int(config["training"]["episodes"]) != 100:
        raise ValueError(
            "legacy training episode default must remain 100 for compatibility"
        )
    method_episodes = config["training"].get("main_method_episodes", {})
    if set(method_episodes) != {
        "UAMCO-DAG",
        "MAPPO",
        "HAPPO",
        "AMCoEdge",
        "FDEdge",
        "MEC-UARA",
    }:
        raise ValueError(
            "training.main_method_episodes must cover all formal methods"
        )
    if any(int(method_episodes[method]) != 300 for method in method_episodes):
        raise ValueError(
            "the fixed-objective protocol requires 300 episodes for every method"
        )
    if int(config["training"].get("ablation_episodes", 0)) != 300:
        raise ValueError(
            "formal causal counterfactuals require the same 300-episode budget "
            "as the complete method"
        )
    seeds = list(config["training"]["seeds"])
    if len(seeds) < 7 or len(set(seeds)) != len(seeds):
        raise ValueError("formal training protocol requires at least seven distinct paired seeds")
    timing = config["timing"]
    calibration = timing.get("calibration", {})
    if float(calibration.get("reference_micro_slot_s", 0.0)) != 0.1:
        raise ValueError("time-resolution reference must be 0.1 seconds")
    if float(calibration.get("configured_micro_slot_s", 0.0)) != float(
        timing.get("micro_slot_s", -1.0)
    ):
        raise ValueError("configured micro-slot must match the calibrated selection")
    if list(calibration.get("candidate_micro_slots_s", ())) != [0.25, 0.5, 1.0]:
        raise ValueError("time-resolution candidates must be 0.25, 0.5, and 1.0 seconds")
    if float(calibration.get("max_primary_metric_relative_error", -1.0)) != 0.01:
        raise ValueError("time-resolution tolerance must be one percent")
    experiments = config.get("experiments", {})
    main_completion = experiments.get("main_completion", {})
    fixed_horizon = experiments.get("fixed_horizon_load", {})
    if (
        main_completion.get("termination") != "fixed_horizon"
        or not bool(main_completion.get("stop_new_admissions"))
        or not bool(main_completion.get("report_remaining"))
    ):
        raise ValueError(
            "main experiment must stop admissions, use a fixed horizon, and report remaining work"
        )
    for key in ("max_resource_utilization", "max_assignment_utilization"):
        value = float(main_completion.get(key, 0.0))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"experiments.main_completion.{key} must lie in (0, 1]")
    minimum_contact = float(main_completion.get("minimum_rsu_contact_ratio", -1.0))
    maximum_contact = float(main_completion.get("maximum_rsu_contact_ratio", -1.0))
    if not 0.0 < minimum_contact < maximum_contact < 1.0:
        raise ValueError(
            "main-completion RSU contact bounds must define a nonzero intermittent range"
        )
    if (
        fixed_horizon.get("termination") != "fixed_horizon"
        or not bool(fixed_horizon.get("report_remaining"))
    ):
        raise ValueError("load experiment must use a fixed horizon and report remaining work")
    objective = config.get("objective", {})
    if objective.get("name") != "delivery_first_edp":
        raise ValueError("the formal protocol requires objective.name=delivery_first_edp")
    interface_scalar = float(objective.get("interface_scalar", math.nan))
    if not math.isfinite(interface_scalar) or not math.isclose(
        interface_scalar, 0.5, rel_tol=0.0, abs_tol=0.0
    ):
        raise ValueError("objective.interface_scalar must be the fixed value 0.5")
    if "preference" in config:
        raise ValueError("preference grids are not part of the fixed delivery-first protocol")
    gpu_ids = list(config["runtime"]["gpu_ids"])
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU identifiers must be unique")
    resources = config["resources"]
    cpu = resources.get("executor_cpu_hz", {})
    if set(cpu) != {"ugv", "rsu", "uav"} or any(float(value) <= 0 for value in cpu.values()):
        raise ValueError("positive UGV/RSU/UAV CPU capacities are required")
    if float(resources.get("link_rate_bps", 0)) <= 0:
        raise ValueError("link rate must be positive")
    mobile_budgets = resources.get("mobile_energy_budget_j", {})
    if set(mobile_budgets) != {"ugv", "uav"} or any(
        float(value) <= 0 or not math.isfinite(float(value))
        for value in mobile_budgets.values()
    ):
        raise ValueError("finite positive UGV/UAV mobile energy budgets are required")
    channel = resources.get("channel", {})
    if any(float(channel.get(key, 0)) <= 0 for key in (
        "bandwidth_hz",
        "noise_interference_w",
        "reference_gain",
        "ground_pathloss_exponent",
        "air_pathloss_exponent",
    )):
        raise ValueError("all physical channel parameters must be positive")
    layouts = config.get("evaluation", {}).get("studies", {}).get("scalability_layouts", ())
    if not layouts or any(
        any(int(layout.get(key, 0)) <= 0 for key in ("ugv_count", "rsu_count", "uav_count"))
        for layout in layouts
    ):
        raise ValueError("scalability study requires positive UGV/RSU/UAV layouts")
    deadline_scales = list(
        config.get("evaluation", {})
        .get("studies", {})
        .get("deadline_multiplier_scale", ())
    )
    if deadline_scales != [1.0, 0.75, 0.5]:
        raise ValueError(
            "constraint stress requires deadline scales [1.0, 0.75, 0.5]"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    config = _read_yaml(Path(path), set())
    validate_config(config)
    return config
