from __future__ import annotations

import bisect
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from .baseline_models import build_adapted_baseline_policy
from .calibration import (
    load_calibration_contract,
    load_frozen_objective_bounds,
)
from .candidate_selection import select_fixed_candidate
from .constrained_ppo import (
    ConstrainedPPOTrainer,
    LagrangeController,
    compute_gae,
)
from .connectivity import CausalContactPredictor
from .env import SLA_TIERS, UAMCOEnv
from .experiment_matrix import (
    CHECKPOINT_SCHEMA_VERSION,
    SEMANTIC_CONTRACT_FILES,
    atomic_write_status,
    build_semantic_contract,
)
from .metrics import (
    ParetoPoint,
    WorkflowOutcome,
    compute_episode_metrics,
    pareto_point_from_episode_metrics,
    pareto_summary,
)
from .objectives import (
    FiniteHorizonObjectiveAccumulator,
    bounded_unit,
    completion_dominance_coefficient,
    delivery_first_episode_cost,
)
from .mobility_data import (
    load_m2dgr_outdoor_traces,
    load_mobility_manifest,
    load_rellis_traces,
    mirror_extend_measured_trace,
)
from .model import HierarchicalConstrainedPolicy
from .observations import (
    EXECUTOR_FEATURE_DIM,
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    GraphObservation,
    batch_graph_observations,
    build_graph_observation,
)
from .physics import shannon_rate_bps
from .sota import (
    SOTAAction,
    SOTAObservation,
    SOTAPhysicalContext,
    SOTATransition,
    build_sota_runtime,
)
from .workflow_data import (
    sample_stratified_workflows,
    stratify_workflows_by_work,
)

from .types import MobilityTrace, WorkflowInstance


DOMAIN_SOTA_METHODS = ("AMCoEdge", "FDEdge", "MEC-UARA")


def requires_causal_result_return(
    *,
    is_sink: bool,
    feeds_high_fan_in_sink: bool,
) -> bool:
    """Whether a remote action must reserve contact through result delivery."""
    return bool(is_sink or feeds_high_fan_in_sink)


def finite_horizon_episode_return(
    *,
    normalized_delay: float,
    normalized_energy: float,
    actual_delivery_progress: float,
    failed_count: int,
    admitted_count: int,
) -> float:
    """Return the negative fixed delivery-first finite-horizon cost."""
    return -delivery_first_episode_cost(
        normalized_delay=normalized_delay,
        normalized_energy=normalized_energy,
        actual_delivery_progress=actual_delivery_progress,
        failed_count=failed_count,
        admitted_count=admitted_count,
    )


def required_remote_contact_s(
    *,
    is_sink: bool,
    input_transfer_s: float,
    queued_cycles: float,
    task_cycles: float,
    cpu_hz: float,
    output_bytes: int,
    link_rate_bps: float,
) -> float:
    """Reserve input delivery for every task and the full return loop for sinks."""
    input_time = max(0.0, float(input_transfer_s))
    if not is_sink:
        return input_time
    cpu = max(1.0, float(cpu_hz))
    link_rate = max(1.0, float(link_rate_bps))
    return (
        input_time
        + max(0.0, float(queued_cycles)) / cpu
        + max(0.0, float(task_cycles)) / cpu
        + 8.0 * max(0, int(output_bytes)) / link_rate
    )


def capture_rng_state() -> dict[str, object]:
    return {
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    random.setstate(state["python_global"])
    np.random.set_state(state["numpy_global"])
    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, Tensor):
        raise RuntimeError("checkpoint CPU RNG state must be a tensor")
    torch.set_rng_state(
        cpu_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    )
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        normalized_cuda_states = []
        for cuda_state in cuda_states:
            if not isinstance(cuda_state, Tensor):
                raise RuntimeError("checkpoint CUDA RNG state must be a tensor")
            normalized_cuda_states.append(
                cuda_state.detach().to(
                    device="cpu",
                    dtype=torch.uint8,
                ).contiguous()
            )
        torch.cuda.set_rng_state_all(normalized_cuda_states)


def hypervolume_reference_sensitivity(
    points: Sequence[ParetoPoint],
    references: Sequence[Sequence[float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for reference in references:
        normalized_reference = (float(reference[0]), float(reference[1]))
        rows.append(
            {
                "reference": list(normalized_reference),
                "hypervolume": float(
                    pareto_summary(
                        points,
                        reference=normalized_reference,
                    )["hypervolume"]
                ),
            }
        )
    return rows


def formal_method_route(method: str) -> str:
    routes = {
        "UAMCO-DAG": "constrained_ppo",
        "MAPPO": "mappo_adapter",
        "HAPPO": "happo_adapter",
        "AMCoEdge": "amcoedge_dqn",
        "FDEdge": "fdedge_diffusion_sac",
        "MEC-UARA": "mec_uara_primal_dual",
    }
    try:
        return routes[method]
    except KeyError as exc:
        raise ValueError(f"unsupported formal method: {method}") from exc


def evaluation_protocols(
    *, stage: str, config: Mapping
) -> tuple[tuple[str, str, str | None, str], ...]:
    primary_mode = str(config["evaluation"]["primary_experiment_mode"])
    if stage == "ablation":
        return (("RELLIS-3D-test", "rellis", "test", primary_mode),)
    return (
        ("RELLIS-3D-test", "rellis", "test", primary_mode),
        ("M2DGR-Outdoor-zero-shot", "m2dgr", None, primary_mode),
        (
            "RELLIS-3D-load",
            "rellis",
            "test",
            str(config["evaluation"]["load_experiment_mode"]),
        ),
    )


def protocol_output_root(config: Mapping, spec, project_root: str | Path) -> Path:
    """Resolve a protocol-specific root and prevent smoke/formal path aliasing."""
    root = Path(project_root).resolve()
    runtime = config.get("runtime", {})
    formal_root = (root / str(runtime["output_dir"])).resolve()
    if spec.protocol == "formal":
        return formal_root
    smoke_dir = runtime.get("smoke_output_dir")
    if not smoke_dir:
        raise ValueError("smoke protocol requires runtime.smoke_output_dir")
    smoke_root = (root / str(smoke_dir)).resolve()
    reserved_formal_root = (
        root / str(runtime.get("formal_output_dir", "output/formal"))
    ).resolve()
    if smoke_root == reserved_formal_root:
        raise ValueError("smoke and formal output roots must be distinct")
    return smoke_root


def validation_delay_weights(spec, config: Mapping) -> tuple[float, ...]:
    del spec
    return (
        validate_interface_scalar(
            config.get("objective", {}).get("interface_scalar", 0.5),
            context="validation objective.interface_scalar",
        ),
    )


def validate_interface_scalar(value: object, *, context: str) -> float:
    """Validate the legacy shared interface scalar before tensor creation."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"compatibility scalar is not numeric ({context}): {value!r}"
        ) from exc
    if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
        raise ValueError(
            "compatibility scalar must be finite and lie in [0, 1] "
            f"({context}): {value!r}"
        )
    return scalar


def validation_selection_kind(spec) -> str:
    del spec
    return "delivery_first_edp"


def validation_selection_objective(
    *,
    selection_kind: str,
    points: Sequence[ParetoPoint],
    records: Sequence[Mapping[str, float]],
    hypervolume_reference: Sequence[float],
    residual_coefficient: float,
) -> dict[str, float]:
    """Select checkpoints with the same finite-horizon delivery residual as training."""
    if not points or not records or len(points) != len(records):
        raise ValueError("validation selection requires aligned non-empty points and records")
    coefficient = float(residual_coefficient)
    progress_values = [
        float(record["actual_delivery_progress_potential"])
        for record in records
    ]
    if (
        not math.isfinite(coefficient)
        or coefficient < 0.0
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in progress_values)
    ):
        raise ValueError("validation delivery residual inputs are invalid")
    mean_progress = float(mean(progress_values))
    delivery_residual = coefficient * (1.0 - mean_progress)
    if selection_kind == "delivery_first_edp":
        objective_values = [
            float(record["delivery_first_objective"])
            for record in records
        ]
        if not all(math.isfinite(value) for value in objective_values):
            raise ValueError("delivery-first validation objectives must be finite")
        objective = float(mean(objective_values))
        raw_selection_value = objective
        delivery_residual = float(
            mean(
                float(record.get("terminal_delivery_residual_cost", 1.0 - float(record["actual_delivery_progress_potential"])))
                for record in records
            )
        )
    elif selection_kind == "scalarized_objective":
        if len(points) != 1:
            raise ValueError("scalarized validation requires exactly one preference")
        weight = float(records[0]["delay_weight"])
        raw_selection_value = (
            weight * points[0].delay_norm
            + (1.0 - weight) * points[0].energy_norm
        )
        objective = raw_selection_value + delivery_residual
    else:
        raise ValueError(f"unknown validation selection kind: {selection_kind}")
    return {
        "objective": float(objective),
        "selection_value": float(raw_selection_value),
        "mean_actual_delivery_progress": mean_progress,
        "delivery_residual_cost": float(delivery_residual),
    }


def select_update_indices(
    count: int,
    maximum: int,
    rng: random.Random,
) -> list[int]:
    """Return a deterministic bounded PPO sample after full-trajectory accounting."""
    if count < 0:
        raise ValueError("decision count cannot be negative")
    if maximum <= 0:
        raise ValueError("maximum update decisions must be positive")
    if count <= maximum:
        return list(range(count))
    return sorted(rng.sample(range(count), maximum))


@dataclass(frozen=True, slots=True)
class PolicyObservationResult:
    actions: Tensor
    executor_logits: Tensor
    old_log_probability: float
    reward_value: float
    cost_values: dict[str, float]
    macro_logits: Tensor


def sample_policy_observation_batch(
    policy: torch.nn.Module,
    observations: Sequence[GraphObservation],
    *,
    device: torch.device | str,
    deterministic: bool,
) -> tuple[PolicyObservationResult, ...]:
    """Sample executor actions for multiple UGVs with one neural forward call."""
    if not observations:
        return ()
    batch = batch_graph_observations(observations).to(device)
    with torch.no_grad():
        output = policy(
            batch.node_features,
            batch.adjacency,
            batch.node_mask,
            batch.global_features,
            batch.delay_weight,
            executor_features=batch.executor_features,
        )
        logits = output["executor_logits"]
        reward_values = output["reward_value"]
        raw_cost_values = output["cost_values"]
        macro_logits = output["macro_logits"]
        if (
            not isinstance(logits, Tensor)
            or not isinstance(reward_values, Tensor)
            or not isinstance(raw_cost_values, Mapping)
            or not isinstance(macro_logits, Tensor)
        ):
            raise TypeError("policy batch returned invalid actor or critic tensors")
        active_mask = batch.node_mask & batch.decision_mask
        defer_only_mask = torch.zeros_like(batch.action_mask)
        defer_only_mask[..., -1] = True
        safe_action_mask = torch.where(
            active_mask.unsqueeze(-1), batch.action_mask, defer_only_mask
        )
        masked_logits = logits.masked_fill(~safe_action_mask, -torch.inf)
        masked_logits = torch.where(
            active_mask.unsqueeze(-1), masked_logits, torch.zeros_like(masked_logits)
        )
        distribution = Categorical(logits=masked_logits)
        actions = (
            masked_logits.argmax(dim=-1)
            if deterministic
            else distribution.sample()
        )
        node_log_probability = distribution.log_prob(actions).masked_fill(
            ~active_mask, 0.0
        )
        denominator = active_mask.sum(dim=-1).clamp_min(1).to(logits.dtype)
        old_log_probabilities = node_log_probability.sum(dim=-1) / denominator
    results: list[PolicyObservationResult] = []
    for batch_index, observation in enumerate(observations):
        node_count = observation.node_features.shape[1]
        results.append(
            PolicyObservationResult(
                actions=actions[batch_index, :node_count].detach().cpu().unsqueeze(0),
                executor_logits=logits[batch_index, :node_count].detach().cpu(),
                old_log_probability=float(old_log_probabilities[batch_index].item()),
                reward_value=float(reward_values[batch_index].item()),
                cost_values={
                    key: float(value[batch_index].item())
                    for key, value in raw_cost_values.items()
                },
                macro_logits=macro_logits[batch_index].detach(),
            )
        )
    return tuple(results)


def build_policy_optimizer(
    policy,
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
) -> torch.optim.Adam:
    actor_group_builder = getattr(policy, "actor_update_groups", None)
    if callable(actor_group_builder):
        trainable_parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
        trainable_ids = {id(parameter) for parameter in trainable_parameters}
        actor_groups = []
        actor_ids: set[int] = set()
        for index, raw_group in enumerate(actor_group_builder()):
            group = [
                parameter
                for parameter in raw_group
                if parameter.requires_grad and id(parameter) in trainable_ids
            ]
            if not group:
                raise ValueError(f"actor update group {index} has no trainable parameters")
            group_ids = {id(parameter) for parameter in group}
            if actor_ids.intersection(group_ids):
                raise ValueError("actor update groups must be disjoint")
            actor_ids.update(group_ids)
            actor_groups.append(
                {
                    "name": f"actor_{index}",
                    "params": group,
                    "lr": float(actor_learning_rate),
                }
            )
        critic_parameters = [
            parameter for parameter in trainable_parameters if id(parameter) not in actor_ids
        ]
        if not critic_parameters:
            raise ValueError("general MARL policy requires separate centralized critics")
        return torch.optim.Adam(
            (
                *actor_groups,
                {
                    "name": "critic",
                    "params": critic_parameters,
                    "lr": float(critic_learning_rate),
                },
            )
        )

    critic_prefixes = ("reward_critic.", "cost_critics.")
    actor_parameters = []
    critic_parameters = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(critic_prefixes):
            critic_parameters.append(parameter)
        else:
            actor_parameters.append(parameter)
    if not actor_parameters or not critic_parameters:
        raise ValueError("policy optimizer requires trainable actor and critic parameters")
    return torch.optim.Adam(
        (
            {"name": "actor", "params": actor_parameters, "lr": float(actor_learning_rate)},
            {"name": "critic", "params": critic_parameters, "lr": float(critic_learning_rate)},
        )
    )


def recorded_position_at_elapsed(
    trace: MobilityTrace,
    *,
    start_index: int,
    elapsed_s: float,
) -> tuple[float, float]:
    if not 0 <= start_index < len(trace.timestamps_s):
        raise ValueError("trajectory start index is invalid")
    if elapsed_s < 0:
        raise ValueError("trajectory elapsed time cannot be negative")
    target = trace.timestamps_s[start_index] + float(elapsed_s)
    index = bisect.bisect_right(trace.timestamps_s, target, lo=start_index) - 1
    index = min(max(start_index, index), len(trace.positions_xy_m) - 1)
    return trace.positions_xy_m[index]


def critical_path_lower_bound_s(
    instance: WorkflowInstance,
    *,
    max_cpu_hz: float,
    max_link_rate_bps: float,
) -> float:
    if max_cpu_hz <= 0 or max_link_rate_bps <= 0:
        raise ValueError("critical-path capacities must be positive")
    indegree = {task_id: len(task.parents) for task_id, task in instance.tasks.items()}
    ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
    longest_finish: dict[str, float] = {}
    while ready:
        task_id = ready.pop(0)
        task = instance.tasks[task_id]
        start = 0.0
        for parent_id in task.parents:
            edge_bytes = sum(
                size_bytes
                for _, size_bytes in instance.dependency_files(parent_id, task_id)
            )
            arrival = longest_finish[parent_id] + edge_bytes * 8.0 / max_link_rate_bps
            start = max(start, arrival)
        longest_finish[task_id] = start + task.cycles / max_cpu_hz
        for child_id in task.children:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                bisect.insort(ready, child_id)
    if len(longest_finish) != len(instance.tasks):
        raise ValueError("critical-path calculation requires a valid DAG")
    return max(longest_finish.values(), default=0.0)


def workflow_resource_lower_bounds_s(
    instance: WorkflowInstance,
    *,
    max_cpu_hz: float,
    fair_share_cpu_hz: float,
    max_link_rate_bps: float,
    assignment_slot_s: float = 0.0,
) -> dict[str, float]:
    if fair_share_cpu_hz <= 0:
        raise ValueError("fair-share compute capacity must be positive")
    if assignment_slot_s < 0 or not math.isfinite(float(assignment_slot_s)):
        raise ValueError("assignment slot must be finite and nonnegative")
    critical_path_s = critical_path_lower_bound_s(
        instance,
        max_cpu_hz=max_cpu_hz,
        max_link_rate_bps=max_link_rate_bps,
    )
    fair_share_work_s = (
        sum(float(task.cycles) for task in instance.tasks.values())
        / fair_share_cpu_hz
    )
    produced_files = {
        file_id
        for task in instance.tasks.values()
        for file_id in task.output_files
    }
    external_input_bytes = sum(
        int(instance.file_sizes[file_id])
        for task in instance.tasks.values()
        for file_id in task.input_files
        if file_id not in produced_files
    )
    sink_output_bytes = sum(
        int(instance.file_sizes[file_id])
        for task_id in instance.sink_task_ids()
        for file_id in instance.tasks[task_id].output_files
    )
    unavoidable_communication_s = (
        (external_input_bytes + sink_output_bytes)
        * 8.0
        / max_link_rate_bps
    )
    assignment_s = len(instance.tasks) * float(assignment_slot_s)
    return {
        "critical_path_s": float(critical_path_s),
        "fair_share_work_s": float(fair_share_work_s),
        "unavoidable_communication_s": float(unavoidable_communication_s),
        "assignment_s": float(assignment_s),
        "combined_s": float(
            max(
                critical_path_s,
                fair_share_work_s,
                unavoidable_communication_s,
                assignment_s,
            )
        ),
    }


def workflow_accessible_fair_share_cpu_hz(
    config: Mapping,
    *,
    experiment_mode: str = "main_completion",
) -> float:
    """Return the CPU share physically accessible to one admitted DAG."""
    scenario = config["scenario"]
    cpu = config["resources"]["executor_cpu_hz"]
    active_workflows = int(
        config.get("experiments", {})
        .get(str(experiment_mode), {})
        .get(
            "active_workflows_per_episode",
            scenario["active_workflows_per_episode"],
        )
    )
    if active_workflows <= 0:
        raise ValueError("active workflow count must be positive")
    shared_cpu_hz = (
        int(scenario["rsu_count"]) * float(cpu["rsu"])
        + int(scenario["uav_count"]) * float(cpu["uav"])
    )
    accessible_cpu_hz = (
        active_workflows * float(cpu["ugv"])
        + shared_cpu_hz
    )
    return float(accessible_cpu_hz / active_workflows)


def workflow_completion_feasibility_s(
    instance: WorkflowInstance,
    *,
    max_cpu_hz: float,
    fair_share_cpu_hz: float,
    max_link_rate_bps: float,
    assignment_slot_s: float,
) -> dict[str, float]:
    """Conservative operational duration for the fixed-horizon experiment."""
    bounds = workflow_resource_lower_bounds_s(
        instance,
        max_cpu_hz=max_cpu_hz,
        fair_share_cpu_hz=fair_share_cpu_hz,
        max_link_rate_bps=max_link_rate_bps,
        assignment_slot_s=assignment_slot_s,
    )
    compute_path_s = max(
        bounds["critical_path_s"],
        bounds["fair_share_work_s"],
    )
    return {
        **bounds,
        "compute_path_s": float(compute_path_s),
        "completion_feasibility_s": float(
            bounds["assignment_s"]
            + compute_path_s
            + bounds["unavoidable_communication_s"]
        ),
    }


def select_workflows_for_experiment(
    workflows: Sequence[WorkflowInstance],
    *,
    config: Mapping,
    feasibility_config: Mapping | None = None,
    experiment_mode: str,
) -> tuple[WorkflowInstance, ...]:
    """Apply the preregistered finite-horizon feasibility contract.

    The completion experiment must contain DAGs that can physically finish
    within its fixed horizon. The load experiment intentionally retains the
    complete production range, including DAGs that are expected to remain.
    """
    items = tuple(workflows)
    if not items:
        return ()
    if experiment_mode != "main_completion":
        return items

    contract_config = feasibility_config or config
    experiment = contract_config["experiments"]["main_completion"]
    resource_utilization = float(experiment["max_resource_utilization"])
    assignment_utilization = float(experiment["max_assignment_utilization"])
    if not 0.0 < resource_utilization <= 1.0:
        raise ValueError("main-completion resource utilization must lie in (0, 1]")
    if not 0.0 < assignment_utilization <= 1.0:
        raise ValueError("main-completion assignment utilization must lie in (0, 1]")

    timing = contract_config["timing"]
    resources = contract_config["resources"]
    episode_s = float(timing["episode_s"])
    decision_slots = math.floor(episode_s / float(timing["micro_slot_s"]))
    maximum_tasks = math.floor(decision_slots * assignment_utilization)
    cpu = resources["executor_cpu_hz"]
    fair_share_cpu_hz = workflow_accessible_fair_share_cpu_hz(contract_config)
    maximum_bound_s = episode_s * resource_utilization

    selected = []
    for instance in items:
        bounds = workflow_completion_feasibility_s(
            instance,
            max_cpu_hz=max(float(value) for value in cpu.values()),
            fair_share_cpu_hz=fair_share_cpu_hz,
            max_link_rate_bps=float(resources["link_rate_bps"]),
            assignment_slot_s=float(timing["micro_slot_s"]),
        )
        if (
            len(instance.tasks) <= maximum_tasks
            and bounds["completion_feasibility_s"]
            <= maximum_bound_s + 1.0e-12
        ):
            selected.append(instance)
    if not selected:
        raise ValueError(
            "main-completion split has no workflow satisfying the fixed-horizon "
            "resource and assignment feasibility contract"
        )
    return tuple(selected)


def curriculum_stage(
    *,
    completed_episode: int,
    total_episodes: int,
    curriculum: Mapping,
) -> str:
    if total_episodes <= 0:
        raise ValueError("total training episodes must be positive")
    episode = int(completed_episode)
    if not 0 <= episode < int(total_episodes):
        raise ValueError("curriculum episode index is outside the job budget")
    easy_fraction = float(curriculum.get("easy_fraction", 0.30))
    medium_fraction = float(curriculum.get("medium_fraction", 0.70))
    if not 0.0 < easy_fraction < medium_fraction < 1.0:
        raise ValueError("curriculum fractions must satisfy 0 < easy < medium < 1")
    progress = episode / float(total_episodes)
    if progress < easy_fraction:
        return "easy"
    if progress < medium_fraction:
        return "medium"
    return "full"


def validation_episode_seed(
    base_seed: int,
    weight_index: int,
    replicate_index: int,
) -> int:
    if weight_index < 0 or replicate_index < 0:
        raise ValueError("validation seed indices must be nonnegative")
    return (
        int(base_seed) * 1_000_000_000
        + 80_000_000
        + int(replicate_index)
    )


def evaluation_episode_seed(
    base_seed: int,
    *,
    condition_index: int,
    weight_index: int,
    replicate_index: int,
) -> int:
    """Return one common episode realization for every compared preference.

    UAMCO-DAG evaluates all preferences inside one job, while single-weight
    baselines evaluate one preference per job.  Including the local weight
    index in this seed would therefore compare different workflows and
    mobility placements.  ``weight_index`` remains an explicit, validated
    argument so callers cannot accidentally hide the pairing contract.
    """
    if condition_index < 0 or weight_index < 0 or replicate_index < 0:
        raise ValueError("evaluation seed indices must be nonnegative")
    return (
        int(base_seed) * 1_000_000
        + int(condition_index) * 100_000
        + int(replicate_index)
    )


def ppo_epochs_for_method(config: Mapping, method: str) -> int:
    algorithm = config["algorithm"]
    overrides = algorithm.get("ppo_epochs_by_method", {})
    epochs = int(overrides.get(str(method), algorithm["ppo_epochs"]))
    if epochs <= 0:
        raise ValueError("PPO epoch count must be positive")
    return epochs


def actual_delivery_progress_potential(env: UAMCOEnv) -> float:
    """Return progress from executed work, dependency delivery, and final return."""
    if not env.workflow_states:
        return 0.0
    workflow_progress: list[float] = []
    for workflow_id, state in env.workflow_states.items():
        total_cycles = sum(
            float(task.cycles) for task in state.instance.tasks.values()
        )
        executed_cycles = sum(
            float(state.instance.tasks[task_id].cycles)
            for task_id in state.compute_completed
        )
        for item in env.running_compute.values():
            if item.workflow_id == workflow_id:
                executed_cycles += float(item.total_cycles - item.remaining_cycles)
        compute_fraction = (
            1.0
            if total_cycles <= 0
            else min(1.0, max(0.0, executed_cycles / total_cycles))
        )

        dependency_bytes = 0
        delivered_dependency_bytes = 0
        for child_id, child in state.instance.tasks.items():
            executor = state.scheduled.get(child_id)
            for parent_id in child.parents:
                for file_id, size_bytes in state.instance.dependency_files(
                    parent_id, child_id
                ):
                    dependency_bytes += int(size_bytes)
                    if (
                        executor is not None
                        and state.file_ledger.has(file_id, executor)
                    ):
                        delivered_dependency_bytes += int(size_bytes)
        delivery_fraction = (
            1.0
            if dependency_bytes <= 0
            else min(
                1.0,
                max(0.0, delivered_dependency_bytes / dependency_bytes),
            )
        )
        final_bytes = 0
        returned_final_bytes = 0
        for sink_id in state.instance.sink_task_ids():
            sink = state.instance.tasks[sink_id]
            for file_id in sink.output_files:
                size_bytes = int(state.instance.file_sizes[file_id])
                final_bytes += size_bytes
                if state.file_ledger.has(file_id, state.owner_ugv):
                    returned_final_bytes += size_bytes
        final_return_fraction = (
            1.0
            if final_bytes <= 0 and state.status == "completed"
            else (
                0.0
                if final_bytes <= 0
                else min(1.0, max(0.0, returned_final_bytes / final_bytes))
            )
        )
        components = [compute_fraction]
        if dependency_bytes > 0:
            components.append(delivery_fraction)
        if final_bytes > 0:
            components.append(final_return_fraction)
        workflow_progress.append(float(sum(components) / len(components)))
    return float(sum(workflow_progress) / len(workflow_progress))


def causal_observation_features_enabled(method: str, variant: str | None) -> bool:
    """Return whether the policy receives the registered causal observations."""
    return (
        str(method) == "UAMCO-DAG"
        and variant != "without_causal_contact_history"
    )


def file_endpoint_refinement_enabled(method: str, variant: str | None) -> bool:
    """Return whether real file/compute custody may refine a macro target."""
    return causal_observation_features_enabled(method, variant) and variant not in {
        "same_state_ppo",
        "macro_only",
    }


def idle_uav_deficit_support_enabled(method: str, variant: str | None) -> bool:
    """Return whether idle UAVs may support a finite-horizon service deficit."""
    return file_endpoint_refinement_enabled(method, variant) and variant != (
        "file_endpoint_only"
    )


def idle_uav_deficit_update_due(*, step_index: int, macro_steps: int) -> bool:
    """Limit whole-DAG service-deficit scans to macro-slot boundaries."""
    interval = int(macro_steps)
    if interval <= 0:
        raise ValueError("macro step interval must be positive")
    return int(step_index) % interval == 0


def finish_time_guard_enabled(method: str, variant: str | None) -> bool:
    """Return whether dominated executor choices receive a physical-time correction."""
    return causal_observation_features_enabled(method, variant) and variant not in {
        "same_state_ppo",
        "without_finish_time_guard",
    }


def delivery_guard_enabled(method: str, variant: str | None) -> bool:
    """Return whether protected causal delivery may defer a macro retargeting."""
    return causal_observation_features_enabled(method, variant) and variant not in {
        "same_state_ppo",
        "macro_only",
    }


def causal_policy_priors_enabled(method: str, variant: str | None) -> bool:
    """Return whether fixed causal residual logits augment the learned policy."""
    return causal_observation_features_enabled(method, variant) and variant != (
        "same_state_ppo"
    )


def predicted_executor_finish_time_s(
    *,
    input_transfer_s: float,
    queued_cycles: float,
    task_cycles: float,
    cpu_hz: float,
    output_return_s: float,
) -> float:
    """Predict physical service time from current state without clipping."""
    cpu = float(cpu_hz)
    if not math.isfinite(cpu) or cpu <= 0.0:
        raise ValueError("executor CPU frequency must be finite and positive")
    components = (
        float(input_transfer_s),
        float(queued_cycles),
        float(task_cycles),
        float(output_return_s),
    )
    if any(math.isnan(value) or value < 0.0 for value in components):
        raise ValueError("finish-time components must be nonnegative and not NaN")
    return float(
        components[0]
        + (components[1] + components[2]) / cpu
        + components[3]
    )


def remaining_workflow_budget_s(
    *,
    current_time_s: float,
    workflow_deadline_s: float,
    episode_end_s: float,
) -> float:
    """Return time remaining before the earlier workflow/episode endpoint."""
    current = float(current_time_s)
    deadline = float(workflow_deadline_s)
    episode_end = float(episode_end_s)
    if not math.isfinite(current) or not math.isfinite(episode_end):
        raise ValueError("current time and episode endpoint must be finite")
    if math.isnan(deadline):
        raise ValueError("workflow deadline must not be NaN")
    return float(max(0.0, min(deadline, episode_end) - current))


def guard_executor_action(
    *,
    selected_action: str,
    legal_actions: Sequence[str],
    predicted_finish_times_s: Mapping[str, float],
    remaining_budget_s: float,
    macro_interval_s: float,
) -> str:
    """Correct only executor choices dominated by the remaining time budget."""
    ordered_legal = tuple(dict.fromkeys(str(action) for action in legal_actions))
    if selected_action not in ordered_legal:
        raise ValueError("selected executor action must be legal")
    budget = max(0.0, float(remaining_budget_s))
    macro_interval = max(0.0, float(macro_interval_s))
    executable = tuple(action for action in ordered_legal if action != "defer")
    if not executable:
        return "defer"

    def finish_time(action: str) -> float:
        value = float(predicted_finish_times_s.get(action, math.inf))
        return value if not math.isnan(value) and value >= 0.0 else math.inf

    fastest = min(executable, key=lambda action: (finish_time(action), ordered_legal.index(action)))
    fastest_time = finish_time(fastest)
    if selected_action == "defer":
        return (
            fastest
            if budget <= fastest_time + macro_interval
            else "defer"
        )
    if finish_time(selected_action) <= budget:
        return selected_action
    feasible = tuple(action for action in executable if finish_time(action) <= budget)
    if feasible:
        return min(
            feasible,
            key=lambda action: (finish_time(action), ordered_legal.index(action)),
        )
    return fastest


def graph_observation_executor_context(
    runtime_context: Mapping[str, object],
) -> dict[str, object]:
    """Strip runtime-only finish-time data from the fixed neural interface."""
    keys = (
        "executor_link_rates_bps",
        "executor_queue_pressures",
        "executor_contact_margins",
        "executor_delivery_feasible",
        "enable_causal_contact_features",
    )
    return {key: runtime_context[key] for key in keys}


def executor_backlog_seconds(env: UAMCOEnv) -> dict[str, float]:
    """Return current serial compute backlog per executor without clipping."""
    result: dict[str, float] = {}
    for executor, cpu_hz in getattr(env, "executor_cpu_hz", {}).items():
        cpu = float(cpu_hz)
        if cpu <= 0.0:
            raise ValueError("executor CPU frequency must be positive")
        queue = getattr(env, "compute_queues", {}).get(executor)
        cycles = sum(
            max(0.0, float(getattr(item, "remaining_cycles", 0.0)))
            for item in (queue.items() if queue is not None else ())
        )
        running = getattr(env, "running_compute", {}).get(executor)
        if running is not None:
            cycles += max(0.0, float(getattr(running, "remaining_cycles", 0.0)))
        result[str(executor)] = float(cycles / cpu)
    return result


def causal_contact_features_enabled(method: str, variant: str | None) -> bool:
    """Compatibility alias for the causal-observation feature contract."""
    return causal_observation_features_enabled(method, variant)


def zone_delivery_profile(
    workflow_states: Mapping[str, object],
    ugv_positions: Mapping[str, tuple[float, float]],
    *,
    width_m: float,
    height_m: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    """Return normalized delivery demand and its live centroid per macro zone."""
    width = float(width_m)
    height = float(height_m)
    if width <= 0 or height <= 0:
        raise ValueError("scenario dimensions must be positive")
    demand = [0.0] * 9
    weighted_x = [0.0] * 9
    weighted_y = [0.0] * 9
    for state in workflow_states.values():
        if getattr(state, "status", None) != "active":
            continue
        owner = str(getattr(state, "owner_ugv"))
        if owner not in ugv_positions:
            continue
        instance = getattr(state, "instance")
        completed = set(getattr(state, "completed", ()))
        ledger = getattr(state, "file_ledger")
        missing_bytes = 0
        for parent_id, child_id in instance.edges:
            if child_id in completed:
                continue
            for file_id, size_bytes in instance.dependency_files(
                parent_id,
                child_id,
            ):
                if not ledger.has(file_id, owner):
                    missing_bytes += int(size_bytes)
        x, y = ugv_positions[owner]
        column = min(2, max(0, int(3.0 * float(x) / width)))
        row = min(2, max(0, int(3.0 * float(y) / height)))
        zone = row * 3 + column
        weight = 1.0 + math.log1p(missing_bytes)
        demand[zone] += weight
        weighted_x[zone] += weight * float(x)
        weighted_y[zone] += weight * float(y)
    targets = tuple(
        (
            (column + 0.5) * width / 3.0,
            (row + 0.5) * height / 3.0,
        )
        if demand[row * 3 + column] <= 0.0
        else (
            weighted_x[row * 3 + column] / demand[row * 3 + column],
            weighted_y[row * 3 + column] / demand[row * 3 + column],
        )
        for row in range(3)
        for column in range(3)
    )
    total = sum(demand)
    if total <= 0:
        return (0.0,) * 9, targets
    return tuple(value / total for value in demand), targets


def zone_delivery_demand(
    workflow_states: Mapping[str, object],
    ugv_positions: Mapping[str, tuple[float, float]],
    *,
    width_m: float,
    height_m: float,
) -> tuple[float, ...]:
    """Return causal current-time delivery demand for the nine macro zones."""
    demand, _ = zone_delivery_profile(
        workflow_states,
        ugv_positions,
        width_m=width_m,
        height_m=height_m,
    )
    return demand


def zone_delivery_targets(
    workflow_states: Mapping[str, object],
    ugv_positions: Mapping[str, tuple[float, float]],
    *,
    width_m: float,
    height_m: float,
) -> tuple[tuple[float, float], ...]:
    """Return a live causal target inside each zone, not its fixed centre."""
    _, targets = zone_delivery_profile(
        workflow_states,
        ugv_positions,
        width_m=width_m,
        height_m=height_m,
    )
    return targets


def causal_uav_service_targets(
    env: UAMCOEnv,
    ugv_positions: Mapping[str, tuple[float, float]],
    *,
    uav_ids: Sequence[str] | None = None,
    current_time_s: float | None = None,
    episode_end_s: float | None = None,
    enable_file_endpoint: bool = True,
    enable_idle_deficit: bool = True,
) -> dict[str, tuple[float, float]]:
    """Refine regional targets using real custody and finite-horizon demand.

    A zone-level macro decision identifies where service is useful, but it
    cannot substitute one UAV for another: bytes queued on ``uav-2`` are not
    deliverable merely because ``uav-0`` covers the same region.  This
    low-level causal controller first binds physical data/compute custody to
    UAV motion, then assigns otherwise idle UAVs only to positive service
    deficits.  It reads current state and never changes simulator state.
    """
    if not enable_file_endpoint:
        return {}
    transfers = [
        item
        for queues in (
            getattr(env, "upload_queues", {}),
            getattr(env, "return_queues", {}),
        )
        for queue in queues.values()
        for item in queue.items()
    ]
    transfers.extend(getattr(env, "waiting_forward_transfers", ()))
    unique: dict[str, object] = {}
    for item in transfers:
        item_id = str(getattr(item, "item_id", id(item)))
        unique.setdefault(item_id, item)

    grouped: dict[str, dict[str, dict[str, float]]] = {}
    bound_workflows: dict[str, str] = {}

    def add_binding(
        *,
        uav_id: str,
        ugv_id: str,
        tier: str,
        deadline: float,
        remaining_work: float,
        workflow_id: str = "",
    ) -> None:
        if ugv_id not in ugv_positions:
            return
        tier_index = SLA_TIERS.index(tier) if tier in SLA_TIERS else len(SLA_TIERS)
        record = grouped.setdefault(uav_id, {}).setdefault(
            ugv_id,
            {
                "tier_index": float(tier_index),
                "deadline_s": float(deadline),
                "remaining_work": 0.0,
                "workflow_id": workflow_id,
            },
        )
        record["tier_index"] = min(record["tier_index"], float(tier_index))
        record["deadline_s"] = min(record["deadline_s"], float(deadline))
        record["remaining_work"] += max(0.0, float(remaining_work))
        if workflow_id and not str(record.get("workflow_id", "")):
            record["workflow_id"] = workflow_id

    for item in unique.values():
        source = str(getattr(item, "source", ""))
        destination = str(getattr(item, "destination", ""))
        if source.startswith("uav-") and destination.startswith("ugv-"):
            uav_id, ugv_id = source, destination
        elif destination.startswith("uav-") and source.startswith("ugv-"):
            uav_id, ugv_id = destination, source
        else:
            continue
        tier = str(getattr(item, "sla_tier", SLA_TIERS[-1]))
        deadline = float(getattr(item, "deadline_s", math.inf))
        remaining_bytes = max(0.0, float(getattr(item, "remaining_bytes", 0.0)))
        add_binding(
            uav_id=uav_id,
            ugv_id=ugv_id,
            tier=tier,
            deadline=deadline,
            remaining_work=remaining_bytes,
            workflow_id=str(getattr(item, "workflow_id", "")),
        )

    compute_items = [
        item
        for queue in getattr(env, "compute_queues", {}).values()
        for item in queue.items()
    ]
    compute_items.extend(getattr(env, "running_compute", {}).values())
    unique_compute: dict[str, object] = {}
    for item in compute_items:
        unique_compute.setdefault(str(getattr(item, "item_id", id(item))), item)
    workflow_states = getattr(env, "workflow_states", {})
    remaining_compute_by_task: dict[tuple[str, str], float] = {}
    for item in unique_compute.values():
        workflow_id = str(getattr(item, "workflow_id", ""))
        task_id = str(getattr(item, "task_id", ""))
        remaining_cycles = max(
            0.0,
            float(getattr(item, "remaining_cycles", 0.0)),
        )
        if workflow_id and task_id:
            remaining_compute_by_task[(workflow_id, task_id)] = remaining_cycles
        executor = str(getattr(item, "executor", ""))
        state = workflow_states.get(workflow_id)
        if not executor.startswith("uav-") or state is None:
            continue
        add_binding(
            uav_id=executor,
            ugv_id=str(getattr(state, "owner_ugv", "")),
            tier=str(getattr(state, "sla_tier", getattr(item, "sla_tier", "bronze"))),
            deadline=float(
                getattr(state, "deadline_time_s", getattr(item, "deadline_s", math.inf))
            ),
            remaining_work=remaining_cycles,
            workflow_id=workflow_id,
        )

    targets: dict[str, tuple[float, float]] = {}
    for uav_id, candidates in grouped.items():
        recipient = min(
            candidates,
            key=lambda ugv_id: (
                candidates[ugv_id]["tier_index"],
                candidates[ugv_id]["deadline_s"],
                -candidates[ugv_id]["remaining_work"],
                str(candidates[ugv_id].get("workflow_id", "")),
                ugv_id,
            ),
        )
        targets[uav_id] = tuple(map(float, ugv_positions[recipient]))
        selected_workflow = str(candidates[recipient].get("workflow_id", ""))
        if selected_workflow:
            bound_workflows[uav_id] = selected_workflow

    if not enable_idle_deficit or uav_ids is None:
        return targets

    now = float(
        getattr(env, "current_time_s", 0.0)
        if current_time_s is None
        else current_time_s
    )
    episode_end = float(
        getattr(env, "episode_s", math.inf)
        if episode_end_s is None
        else episode_end_s
    )
    executor_cpu_hz = {
        str(executor): max(0.0, float(cpu_hz))
        for executor, cpu_hz in getattr(env, "executor_cpu_hz", {}).items()
    }
    deficits: dict[str, dict[str, object]] = {}
    for workflow_id, state in workflow_states.items():
        if getattr(state, "status", None) != "active":
            continue
        owner = str(getattr(state, "owner_ugv", ""))
        if owner not in ugv_positions:
            continue
        completed = set(
            getattr(state, "compute_completed", getattr(state, "completed", ()))
        )
        remaining_cycles = 0.0
        for task_id, task in getattr(state.instance, "tasks", {}).items():
            if task_id in completed:
                continue
            remaining_cycles += remaining_compute_by_task.get(
                (str(workflow_id), str(task_id)),
                max(0.0, float(getattr(task, "cycles", 0.0))),
            )
        budget = remaining_workflow_budget_s(
            current_time_s=now,
            workflow_deadline_s=float(getattr(state, "deadline_time_s", episode_end)),
            episode_end_s=episode_end,
        )
        required_rate = (
            math.inf if budget <= 0.0 and remaining_cycles > 0.0
            else remaining_cycles / max(budget, 1.0e-12)
        )
        bound_rate = executor_cpu_hz.get(owner, 0.0) + sum(
            executor_cpu_hz.get(uav_id, 0.0)
            for uav_id, bound_workflow in bound_workflows.items()
            if bound_workflow == str(workflow_id)
        )
        deficit = max(0.0, required_rate - bound_rate)
        if deficit <= 0.0:
            continue
        tier = str(getattr(state, "sla_tier", SLA_TIERS[-1]))
        deficits[str(workflow_id)] = {
            "owner": owner,
            "tier_index": (
                SLA_TIERS.index(tier) if tier in SLA_TIERS else len(SLA_TIERS)
            ),
            "deadline_s": float(getattr(state, "deadline_time_s", episode_end)),
            "deficit": deficit,
        }

    idle_uavs = tuple(sorted(set(map(str, uav_ids)) - set(targets)))
    for uav_id in idle_uavs:
        positive = tuple(
            workflow_id
            for workflow_id, record in deficits.items()
            if float(record["deficit"]) > 0.0
        )
        if not positive:
            break
        selected_workflow = min(
            positive,
            key=lambda workflow_id: (
                int(deficits[workflow_id]["tier_index"]),
                float(deficits[workflow_id]["deadline_s"]),
                -float(deficits[workflow_id]["deficit"]),
                workflow_id,
                uav_id,
            ),
        )
        owner = str(deficits[selected_workflow]["owner"])
        targets[uav_id] = tuple(map(float, ugv_positions[owner]))
        deficits[selected_workflow]["deficit"] = max(
            0.0,
            float(deficits[selected_workflow]["deficit"])
            - executor_cpu_hz.get(uav_id, 0.0),
        )
    return targets


def resolve_smoke_workflows(fold, config: Mapping) -> tuple[WorkflowInstance, ...]:
    del config
    return tuple(fold.train)


def resolve_smoke_experiment_mode(config: Mapping) -> str:
    return str(
        config.get("training", {}).get(
            "experiment_mode",
            "main_completion",
        )
    )


@dataclass(frozen=True, slots=True)
class TrajectoryPlacement:
    trace: MobilityTrace
    start_index: int
    rotation_rad: float
    translation_xy_m: tuple[float, float]

    def position_at(self, elapsed_s: float) -> tuple[float, float]:
        raw_x, raw_y = recorded_position_at_elapsed(
            self.trace, start_index=self.start_index, elapsed_s=elapsed_s
        )
        cosine = math.cos(self.rotation_rad)
        sine = math.sin(self.rotation_rad)
        translated_x, translated_y = self.translation_xy_m
        return (
            cosine * raw_x - sine * raw_y + translated_x,
            sine * raw_x + cosine * raw_y + translated_y,
        )


@dataclass(frozen=True, slots=True)
class FixedHorizonTraceSelection:
    eligible: tuple[MobilityTrace, ...]
    excluded: tuple[tuple[str, float], ...]


def select_fixed_horizon_traces(
    traces: Sequence[MobilityTrace],
    *,
    episode_s: float,
    minimum_count: int,
) -> FixedHorizonTraceSelection:
    horizon = float(episode_s)
    count = int(minimum_count)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("formal episode horizon must be finite and positive")
    if count <= 0:
        raise ValueError("required measured trajectory count must be positive")
    eligible: list[MobilityTrace] = []
    excluded: list[tuple[str, float]] = []
    for trace in traces:
        if len(trace.timestamps_s) < 2:
            duration = 0.0
        else:
            duration = max(
                0.0,
                float(trace.timestamps_s[-1]) - float(trace.timestamps_s[0]),
            )
        if duration + 1.0e-9 >= horizon:
            eligible.append(trace)
        else:
            excluded.append((str(trace.trace_id), duration))
    if len(eligible) < count:
        raise ValueError(
            "zero-shot evaluation has "
            f"{len(eligible)} eligible measured trajectories; at least "
            f"{count} are required at the {horizon:g}-second formal horizon"
        )
    return FixedHorizonTraceSelection(
        eligible=tuple(eligible),
        excluded=tuple(excluded),
    )


@dataclass(slots=True)
class RolloutDecision:
    observation: GraphObservation
    actions: Tensor
    old_log_probability: float
    reward_value: float
    cost_values: dict[str, float]
    macro_action: int | None = None
    macro_old_log_probability: float = 0.0
    reward: float = 0.0
    costs: dict[str, float] | None = None


def evaluate_rollout_decision_batch(
    policy: torch.nn.Module,
    decisions: Sequence[RolloutDecision],
    *,
    device: torch.device | str,
) -> tuple[Tensor, Tensor, Mapping[str, Tensor | Mapping[str, Tensor]]]:
    """Re-evaluate one PPO chunk with a single padded-graph policy call."""
    if not decisions:
        raise ValueError("cannot evaluate an empty rollout-decision batch")
    observations = tuple(decision.observation for decision in decisions)
    batch = batch_graph_observations(observations).to(device)
    actions = torch.full(
        batch.node_mask.shape,
        -1,
        dtype=torch.long,
        device=device,
    )
    for batch_index, decision in enumerate(decisions):
        node_count = decision.observation.node_features.shape[1]
        if decision.actions.shape != (1, node_count):
            raise ValueError("recorded rollout action shape does not match its graph")
        actions[batch_index, :node_count] = decision.actions[0].to(device)

    log_probability, entropy, output = policy.evaluate_actions(
        batch.node_features,
        batch.adjacency,
        batch.node_mask,
        batch.global_features,
        batch.delay_weight,
        decision_mask=batch.decision_mask,
        actions=actions,
        action_mask=batch.action_mask,
        executor_features=batch.executor_features,
    )
    batch_size = len(decisions)
    if log_probability.shape != (batch_size,) or entropy.shape != (batch_size,):
        raise ValueError("policy returned invalid batched PPO actor statistics")
    reward_values = output.get("reward_value")
    cost_values = output.get("cost_values")
    macro_logits = output.get("macro_logits")
    if (
        not isinstance(reward_values, Tensor)
        or reward_values.shape != (batch_size,)
        or not isinstance(cost_values, Mapping)
        or not isinstance(macro_logits, Tensor)
        or macro_logits.shape[0] != batch_size
    ):
        raise ValueError("policy returned invalid batched PPO critic or macro tensors")
    if any(
        not isinstance(value, Tensor) or value.shape != (batch_size,)
        for value in cost_values.values()
    ):
        raise ValueError("policy returned invalid batched PPO cost values")

    macro_presence = torch.tensor(
        [decision.macro_action is not None for decision in decisions],
        dtype=torch.bool,
        device=device,
    )
    if bool(macro_presence.any()):
        macro_actions = torch.tensor(
            [
                0 if decision.macro_action is None else int(decision.macro_action)
                for decision in decisions
            ],
            dtype=torch.long,
            device=device,
        )
        macro_distribution = Categorical(logits=macro_logits)
        log_probability = log_probability + torch.where(
            macro_presence,
            macro_distribution.log_prob(macro_actions),
            torch.zeros_like(log_probability),
        )
        entropy = entropy + torch.where(
            macro_presence,
            macro_distribution.entropy(),
            torch.zeros_like(entropy),
        )
    return log_probability, entropy, output


@dataclass(frozen=True, slots=True)
class EpisodeRollout:
    decisions: tuple[RolloutDecision, ...]
    outcomes: tuple[WorkflowOutcome, ...]
    metrics: dict[str, float]
    admitted_count: int
    episode_return: float
    update_summary: Mapping[str, float] | None = None


class FormalExperimentRunner:
    def __init__(self, *, config: Mapping, spec, project_root: str | Path) -> None:
        self.config = config
        self.spec = spec
        self.project_root = Path(project_root).resolve()
        self.semantic_contract = build_semantic_contract(
            config,
            project_root=self.project_root,
        )
        calibration = config.get("calibration", {})
        if calibration.get("update_during_training") is not False:
            raise ValueError("calibration.update_during_training must be false")
        manifest_path = calibration.get("manifest")
        if not manifest_path:
            raise ValueError("formal runtime requires calibration.manifest")
        frozen_bounds_path = calibration.get("frozen_bounds")
        if not frozen_bounds_path:
            raise ValueError("formal runtime requires calibration.frozen_bounds")
        calibration_manifest = (self.project_root / str(manifest_path)).resolve()
        calibration_active_config = config if spec.protocol == "formal" else None
        self.calibration_contract = load_calibration_contract(
            calibration_manifest,
            project_root=self.project_root,
            objective_bounds_path=frozen_bounds_path,
            active_config=calibration_active_config,
        )
        self.objective_bounds = load_frozen_objective_bounds(
            frozen_bounds_path,
            project_root=self.project_root,
            active_config=calibration_active_config,
        )
        self.output_root = protocol_output_root(config, spec, self.project_root)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.device(self.device).type == "cuda":
            memory_fraction = float(
                config.get("runtime", {}).get(
                    "gpu_memory_fraction_per_worker", 1.0
                )
            )
            if not 0.0 < memory_fraction <= 1.0:
                raise ValueError("GPU memory fraction per worker must lie in (0, 1]")
            torch.cuda.set_per_process_memory_fraction(
                memory_fraction,
                device=self.device,
            )
            torch.cuda.reset_peak_memory_stats(self.device)
        self.cost_keys = tuple(
            f"{tier}_{kind}" for tier in SLA_TIERS for kind in ("miss", "drop")
        )
        scale_layouts = config["evaluation"]["studies"]["scalability_layouts"]
        maximum_rsus = max(
            int(config["scenario"]["rsu_count"]),
            *(int(layout["rsu_count"]) for layout in scale_layouts),
        )
        maximum_uavs = max(
            int(config["scenario"]["uav_count"]),
            *(int(layout["uav_count"]) for layout in scale_layouts),
        )
        self.infrastructure_actions = tuple(
            [f"rsu-{index}" for index in range(maximum_rsus)]
            + [f"uav-{index}" for index in range(maximum_uavs)]
        )
        self.executor_actions = ("local", *self.infrastructure_actions, "defer")
        self._unavailable_uav_count = 0
        self._contact_forecast_error = 0.0
        self._deadline_multiplier_scale = 1.0
        self.rng = random.Random(int(spec.seed))
        self._seed_everything(int(spec.seed))
        self.best_validation_objective = math.inf
        self.corpus, self.fold, self.mobility = self._load_assets()
        if self.spec.protocol == "formal":
            self._validate_main_completion_assets()
        self.method_route = formal_method_route(self.spec.method)
        self.is_domain_sota = self.spec.method in DOMAIN_SOTA_METHODS
        maximum_ugvs = max(
            int(config["scenario"]["ugv_count"]),
            *(int(layout["ugv_count"]) for layout in scale_layouts),
        )
        self.sota_runtime = None
        self.policy = None
        self.optimizer = None
        if self.is_domain_sota:
            self.sota_runtime = build_sota_runtime(
                self.spec.method,
                action_count=len(self.executor_actions),
                agent_count=maximum_ugvs,
                device=self.device,
                seed=int(self.spec.seed),
                project_root=self.project_root,
                max_grad_norm=float(self.config["algorithm"]["max_grad_norm"]),
            )
        else:
            self.policy = self._build_policy().to(self.device)
            self.optimizer = build_policy_optimizer(
                self.policy,
                actor_learning_rate=float(config["algorithm"]["actor_learning_rate"]),
                critic_learning_rate=float(config["algorithm"]["critic_learning_rate"]),
            )
        limits = {
            f"{tier}_{kind}": float(config["sla"]["tiers"][tier][f"{kind}_limit"])
            for tier in SLA_TIERS
            for kind in ("miss", "drop")
        }
        self.lagrange = LagrangeController(
            limits=limits,
            learning_rate=float(config["algorithm"]["lagrange_learning_rate"]),
        )
        self.ppo = None if self.is_domain_sota else ConstrainedPPOTrainer(
            self.lagrange,
            clip_ratio=float(config["algorithm"]["clip_ratio"]),
            value_coefficient=float(config["algorithm"]["value_coefficient"]),
            cost_value_coefficient=float(config["algorithm"]["cost_value_coefficient"]),
            entropy_coefficient=float(config["algorithm"]["entropy_coefficient"]),
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _load_assets(self):
        from .workflow_data import build_lofo_folds, load_workflow_corpus

        workflow_manifest = json.loads(
            (self.project_root / self.config["data"]["workflow_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        first_record = next(iter(workflow_manifest["families"].values()))
        pegasus_root = (self.project_root / first_record["path"]).parent
        corpus = load_workflow_corpus(
            pegasus_root,
            reference_frequency_hz=float(self.config["data"]["reference_frequency_hz"]),
        )
        folds = build_lofo_folds(
            corpus,
            excluded_instance_ids=self.calibration_contract.workflow_instance_ids,
        )
        mobility_payload = load_mobility_manifest(
            self.project_root / self.config["data"]["mobility_manifest"]
        )
        rellis = mobility_payload["rellis3d"]
        rellis_traces = load_rellis_traces(
            self.project_root / rellis["local_root"],
            rellis["splits"],
            sampling_hz=float(rellis["sampling_hz"]),
        )
        if tuple(trace.trace_id for trace in rellis_traces.get("calibration", ())) != (
            self.calibration_contract.mobility_trace_id,
        ):
            raise ValueError("formal runtime calibration mobility selection mismatch")
        formal_rellis = {
            split: traces
            for split, traces in rellis_traces.items()
            if split != "calibration"
        }
        if any(
            trace.trace_id == self.calibration_contract.mobility_trace_id
            for traces in formal_rellis.values()
            for trace in traces
        ):
            raise RuntimeError("calibration mobility trace leaked into formal runtime")
        self._m2dgr_manifest = dict(mobility_payload["m2dgr_outdoor"])
        self._m2dgr_cache = None
        self._m2dgr_selection = None
        return corpus, folds[self.spec.fold], {"rellis": formal_rellis}

    def _zero_shot_mobility(self) -> tuple[MobilityTrace, ...]:
        if self._m2dgr_cache is None:
            manifest = self._m2dgr_manifest
            loaded = tuple(
                load_m2dgr_outdoor_traces(
                    self.project_root / str(manifest["local_root"]),
                    manifest["sequences"],
                )
            )
            self._m2dgr_selection = select_fixed_horizon_traces(
                loaded,
                episode_s=float(self.config["timing"]["episode_s"]),
                minimum_count=1,
            )
            self._m2dgr_cache = self._m2dgr_selection.eligible
        return self._m2dgr_cache

    def _validate_main_completion_assets(self) -> None:
        for split_name in ("train", "validation", "test"):
            selected = select_workflows_for_experiment(
                getattr(self.fold, split_name),
                config=self.config,
                experiment_mode="main_completion",
            )
            if not selected:
                raise RuntimeError(
                    f"main-completion {split_name} split is empty after feasibility filtering"
                )

        traces = self.mobility["rellis"]["train"]
        placements = self._trajectory_placements(
            traces,
            rng=random.Random(int(self.spec.seed) * 100_000),
        )
        rsu_positions = self._rsu_positions()
        dt_s = float(self.config["timing"]["micro_slot_s"])
        episode_s = float(self.config["timing"]["episode_s"])
        connected_pairs = 0
        observed_pairs = 0
        for step in range(math.floor(episode_s / dt_s) + 1):
            current_time_s = step * dt_s
            ugv_positions = {
                ugv_id: placement.position_at(current_time_s)
                for ugv_id, placement in placements.items()
            }
            connectivity = self._connectivity(ugv_positions, rsu_positions, {})
            connected_pairs += sum(int(value) for value in connectivity.values())
            observed_pairs += len(connectivity)
        contact_ratio = connected_pairs / max(1, observed_pairs)
        contract = self.config["experiments"]["main_completion"]
        minimum = float(contract["minimum_rsu_contact_ratio"])
        maximum = float(contract["maximum_rsu_contact_ratio"])
        if not minimum <= contact_ratio <= maximum:
            raise RuntimeError(
                "primary RELLIS geometry violates the intermittent-RSU-contact "
                f"contract: observed={contact_ratio:.6f}, required=[{minimum}, {maximum}]"
            )
        self.primary_rsu_contact_ratio = float(contact_ratio)

    def _build_policy(self):
        common = {
            "node_feature_dim": NODE_FEATURE_DIM,
            "global_feature_dim": GLOBAL_FEATURE_DIM,
            "hidden_dim": int(self.config["algorithm"]["graph_hidden_dim"]),
            "num_executor_actions": len(self.executor_actions),
            "num_macro_actions": 9,
            "cost_keys": self.cost_keys,
            "action_feature_dim": EXECUTOR_FEATURE_DIM,
        }
        if self.spec.method in {"MAPPO", "HAPPO"}:
            policy = build_adapted_baseline_policy(self.spec.method, **common)
        elif self.spec.variant == "without_gnn":
            policy = build_adapted_baseline_policy("MAPPO", **common)
        else:
            policy = HierarchicalConstrainedPolicy(
                **common,
                num_uavs=int(self.config["scenario"]["uav_count"]),
                causal_executor_prior_scale=(
                    float(
                        self.config["algorithm"].get(
                            "causal_executor_prior_scale",
                            1.0,
                        )
                    )
                    if causal_policy_priors_enabled(
                        self.spec.method,
                        self.spec.variant,
                    )
                    else 0.0
                ),
                causal_macro_prior_scale=(
                    float(
                        self.config["algorithm"].get(
                            "causal_macro_prior_scale",
                            4.0,
                        )
                    )
                    if causal_policy_priors_enabled(
                        self.spec.method,
                        self.spec.variant,
                    )
                    else 0.0
                ),
                num_graph_layers=(
                    1
                    if self.spec.variant == "non_hierarchical"
                    else int(self.config["algorithm"]["graph_layers"])
                ),
            )
        if self.spec.variant == "without_cost_critics":
            for critic in policy.cost_critics.values():
                torch.nn.init.zeros_(critic.weight)
                torch.nn.init.zeros_(critic.bias)
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
        return policy

    def _new_env(self, *, experiment_mode: str | None = None) -> UAMCOEnv:
        from .allocator import TierAgnosticAllocator

        scenario = self.config["scenario"]
        resources = self.config["resources"]
        cpu = resources["executor_cpu_hz"]
        executors = {
            **{f"ugv-{index}": float(cpu["ugv"]) for index in range(int(scenario["ugv_count"]))},
            **{f"rsu-{index}": float(cpu["rsu"]) for index in range(int(scenario["rsu_count"]))},
            **{f"uav-{index}": float(cpu["uav"]) for index in range(int(scenario["uav_count"]))},
        }
        queue = scenario["queue_capacity"]
        experiment_mode = str(
            experiment_mode
            or self.config["training"].get("experiment_mode", "main_completion")
        )
        termination_mode = str(
            self.config["experiments"][experiment_mode]["termination"]
        )
        episode_s = float(
            self.config["experiments"][experiment_mode].get(
                "episode_s",
                self.config["timing"]["episode_s"],
            )
        )
        return UAMCOEnv(
            micro_slot_s=float(self.config["timing"]["micro_slot_s"]),
            macro_interval_s=float(self.config["timing"]["macro_interval_s"]),
            episode_s=episode_s,
            executor_cpu_hz=executors,
            link_rate_bps=float(resources["link_rate_bps"]),
            ttl_deadline_multiplier=float(self.config["sla"]["ttl_deadline_multiplier"]),
            max_active_workflows_per_ugv=int(queue["ugv_active_dags"]),
            default_queue_items=max(int(queue["rsu_compute_items"]), int(queue["uav_compute_items"])),
            default_buffer_bytes=max(
                int(queue["ugv_buffer_bytes"]),
                int(queue["rsu_buffer_bytes"]),
                int(queue["uav_buffer_bytes"]),
            ),
            compute_capacitance=resources["compute_capacitance"],
            radio_power_w=resources["radio_power_w"],
            mobile_energy_budget_j=resources["mobile_energy_budget_j"],
            allocator=(
                TierAgnosticAllocator()
                if self.spec.variant == "without_sla_allocator"
                else None
            ),
            unlock_on_compute_completion=(
                self.spec.variant == "computation_completion_unlock"
            ),
            proactive_high_fan_in_delivery=delivery_guard_enabled(
                getattr(self.spec, "method", "UAMCO-DAG"),
                getattr(self.spec, "variant", None),
            ),
            termination_mode=termination_mode,
            queue_capacity_by_kind={
                "ugv": {
                    "max_items": max(8, int(queue["ugv_active_dags"])),
                    "max_bytes": int(queue["ugv_buffer_bytes"]),
                },
                "rsu": {
                    "max_items": int(queue["rsu_compute_items"]),
                    "max_bytes": int(queue["rsu_buffer_bytes"]),
                },
                "uav": {
                    "max_items": int(queue["uav_compute_items"]),
                    "max_bytes": int(queue["uav_buffer_bytes"]),
                },
            },
        )

    def _trajectory_placements(
        self,
        traces: Sequence[MobilityTrace],
        *,
        rng: random.Random,
    ) -> dict[str, TrajectoryPlacement]:
        if not traces:
            raise ValueError("an episode requires measured mobility traces")
        episode_s = float(self.config["timing"]["episode_s"])
        ugv_count = int(self.config["scenario"]["ugv_count"])
        grid_columns = math.ceil(math.sqrt(ugv_count))
        grid_rows = math.ceil(ugv_count / grid_columns)
        width = float(self.config["scenario"]["width_m"])
        height = float(self.config["scenario"]["height_m"])
        placements: dict[str, TrajectoryPlacement] = {}
        for ugv_index in range(ugv_count):
            trace = traces[ugv_index % len(traces)]
            if trace.timestamps_s[-1] - trace.timestamps_s[0] < episode_s:
                trace = mirror_extend_measured_trace(
                    trace,
                    minimum_duration_s=episode_s,
                )
            last_start_time = trace.timestamps_s[-1] - episode_s
            last_start_index = bisect.bisect_right(trace.timestamps_s, last_start_time) - 1
            if last_start_index < 0:
                raise ValueError(
                    f"measured trajectory {trace.trace_id} is shorter than one formal episode"
                )
            start_index = rng.randint(0, last_start_index)
            rotation = (ugv_index % 4) * math.pi / 2.0
            raw_x, raw_y = trace.positions_xy_m[start_index]
            rotated_x = math.cos(rotation) * raw_x - math.sin(rotation) * raw_y
            rotated_y = math.sin(rotation) * raw_x + math.cos(rotation) * raw_y
            column = ugv_index % grid_columns
            row = ugv_index // grid_columns
            target = (
                (column + 0.5) * width / grid_columns,
                (row + 0.5) * height / grid_rows,
            )
            placements[f"ugv-{ugv_index}"] = TrajectoryPlacement(
                trace=trace,
                start_index=start_index,
                rotation_rad=rotation,
                translation_xy_m=(target[0] - rotated_x, target[1] - rotated_y),
            )
        return placements

    def _rsu_positions(self) -> dict[str, tuple[float, float]]:
        width = float(self.config["scenario"]["width_m"])
        height = float(self.config["scenario"]["height_m"])
        count = int(self.config["scenario"]["rsu_count"])
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        return {
            f"rsu-{index}": (
                ((index % columns) + 0.5) * width / columns,
                ((index // columns) + 0.5) * height / rows,
            )
            for index in range(count)
        }

    def _initial_uav_positions(self) -> dict[str, tuple[float, float]]:
        width = float(self.config["scenario"]["width_m"])
        height = float(self.config["scenario"]["height_m"])
        count = int(self.config["scenario"]["uav_count"])
        zones = tuple(
            ((column + 0.5) * width / 3.0, (row + 0.5) * height / 3.0)
            for row in range(3)
            for column in range(3)
        )
        return {f"uav-{index}": zones[index % len(zones)] for index in range(count)}

    def _macro_targets(
        self,
        macro_action: int,
        *,
        zone_targets: Sequence[tuple[float, float]] | None = None,
    ) -> dict[str, tuple[float, float]]:
        width = float(self.config["scenario"]["width_m"])
        height = float(self.config["scenario"]["height_m"])
        zones = (
            tuple((float(x), float(y)) for x, y in zone_targets)
            if zone_targets is not None
            else tuple(
                ((column + 0.5) * width / 3.0, (row + 0.5) * height / 3.0)
                for row in range(3)
                for column in range(3)
            )
        )
        if len(zones) != 9:
            raise ValueError("macro targeting requires exactly nine zone targets")
        return {
            f"uav-{index}": zones[(int(macro_action) + index) % len(zones)]
            for index in range(int(self.config["scenario"]["uav_count"]))
        }

    def _move_uavs(
        self,
        positions: Mapping[str, tuple[float, float]],
        targets: Mapping[str, tuple[float, float]],
    ) -> tuple[dict[str, tuple[float, float]], dict[str, float]]:
        dt_s = float(self.config["timing"]["micro_slot_s"])
        maximum_distance = float(self.config["scenario"]["uav_max_speed_mps"]) * dt_s
        updated: dict[str, tuple[float, float]] = {}
        speeds: dict[str, float] = {}
        for uav_id, current in positions.items():
            target = targets.get(uav_id, current)
            distance = math.dist(current, target)
            if distance <= maximum_distance or distance == 0:
                new_position = target
            else:
                ratio = maximum_distance / distance
                new_position = (
                    current[0] + ratio * (target[0] - current[0]),
                    current[1] + ratio * (target[1] - current[1]),
                )
            updated[uav_id] = new_position
            speeds[uav_id] = math.dist(current, new_position) / dt_s
        return updated, speeds

    def _connectivity(
        self,
        ugv_positions: Mapping[str, tuple[float, float]],
        rsu_positions: Mapping[str, tuple[float, float]],
        uav_positions: Mapping[str, tuple[float, float]],
    ) -> dict[tuple[str, str], bool]:
        rsu_radius = float(self.config["scenario"]["rsu_radius_m"])
        uav_radius = float(self.config["scenario"]["uav_radius_m"])
        connectivity: dict[tuple[str, str], bool] = {}
        for ugv_id, ugv_position in ugv_positions.items():
            for rsu_id, position in rsu_positions.items():
                connectivity[(ugv_id, rsu_id)] = math.dist(ugv_position, position) <= rsu_radius
            for uav_id, position in uav_positions.items():
                uav_index = int(uav_id.split("-")[-1])
                unavailable_from = int(self.config["scenario"]["uav_count"]) - self._unavailable_uav_count
                connectivity[(ugv_id, uav_id)] = (
                    uav_index < unavailable_from
                    and math.dist(ugv_position, position) <= uav_radius
                )
        return connectivity

    def _link_rates(
        self,
        ugv_positions: Mapping[str, tuple[float, float]],
        rsu_positions: Mapping[str, tuple[float, float]],
        uav_positions: Mapping[str, tuple[float, float]],
        connectivity: Mapping[tuple[str, str], bool],
    ) -> dict[tuple[str, str], float]:
        resources = self.config["resources"]
        channel = resources["channel"]
        powers = resources["radio_power_w"]
        rates: dict[tuple[str, str], float] = {}
        for (ugv_id, infrastructure_id), connected in connectivity.items():
            if not connected:
                rates[(ugv_id, infrastructure_id)] = 0.0
                continue
            ugv_position = ugv_positions[ugv_id]
            if infrastructure_id.startswith("rsu-"):
                distance = max(1.0, math.dist(ugv_position, rsu_positions[infrastructure_id]))
                exponent = float(channel["ground_pathloss_exponent"])
                destination_kind = "rsu"
            else:
                horizontal = math.dist(ugv_position, uav_positions[infrastructure_id])
                distance = max(
                    1.0,
                    math.hypot(horizontal, float(self.config["scenario"]["uav_altitude_m"])),
                )
                exponent = float(channel["air_pathloss_exponent"])
                destination_kind = "uav"
            transmit_power = min(float(powers["ugv"]), float(powers[destination_kind]))
            signal_power = (
                transmit_power
                * float(channel["reference_gain"])
                / distance**exponent
            )
            rates[(ugv_id, infrastructure_id)] = min(
                float(resources["link_rate_bps"]),
                shannon_rate_bps(
                    float(channel["bandwidth_hz"]),
                    signal_power_w=signal_power,
                    noise_interference_w=float(channel["noise_interference_w"]),
                ),
            )
        return rates

    def _contact_history_score(
        self,
        position_history: Sequence[tuple[float, float]],
        rsu_positions: Mapping[str, tuple[float, float]],
        uav_positions: Mapping[str, tuple[float, float]],
        uav_position_histories: Mapping[
            str, Sequence[tuple[float, float]]
        ] | None = None,
    ) -> float:
        if self.spec.variant == "without_causal_contact_history" or len(position_history) < 2:
            return 0.0
        predictor = CausalContactPredictor(history_length=min(5, len(position_history)))
        horizon = float(self.config["timing"]["macro_interval_s"])
        sample_interval = float(self.config["timing"]["micro_slot_s"])
        viable = 0
        total = 0
        for _, position in rsu_positions.items():
            total += 1
            viable += int(
                predictor.forecast_distance(
                    position_history,
                    infrastructure_xy=position,
                    horizon_s=horizon,
                    sample_interval_s=sample_interval,
                )
                <= float(self.config["scenario"]["rsu_radius_m"])
                * (1.0 - self._contact_forecast_error)
            )
        for uav_id, position in uav_positions.items():
            total += 1
            viable += int(
                predictor.forecast_distance(
                    position_history,
                    infrastructure_xy=position,
                    horizon_s=horizon,
                    sample_interval_s=sample_interval,
                    infrastructure_history_xy=(
                        (uav_position_histories or {}).get(uav_id)
                    ),
                )
                <= float(self.config["scenario"]["uav_radius_m"])
                * (1.0 - self._contact_forecast_error)
            )
        return viable / max(1, total)

    @staticmethod
    def _finite_queue_pressure(queue) -> float:
        if queue is None:
            return 0.0
        item_ratio = 1.0 - queue.remaining_items / queue.max_items
        byte_ratio = (
            0.0
            if queue.max_bytes == 0
            else queue.used_bytes / queue.max_bytes
        )
        return float(min(1.0, max(0.0, max(item_ratio, byte_ratio))))

    @staticmethod
    def _effective_queue_pressure(
        queue,
        *,
        running_item,
        cpu_hz: float,
        horizon_s: float,
    ) -> float:
        """Combine finite-buffer occupancy with predicted compute backlog."""
        finite_pressure = FormalExperimentRunner._finite_queue_pressure(queue)
        cpu = float(cpu_hz)
        horizon = float(horizon_s)
        if queue is None or cpu <= 0.0 or horizon <= 0.0:
            return finite_pressure
        queued_cycles = sum(
            max(0.0, float(getattr(item, "remaining_cycles", 0.0)))
            for item in queue.items()
        )
        if running_item is not None:
            queued_cycles += max(
                0.0,
                float(getattr(running_item, "remaining_cycles", 0.0)),
            )
        workload_pressure = queued_cycles / (cpu * horizon)
        return float(
            min(1.0, max(0.0, finite_pressure, workload_pressure))
        )

    def _executor_observation_context(
        self,
        *,
        env: UAMCOEnv,
        state,
        candidate_task_id: str,
        owner_ugv: str,
        link_rates: Mapping[tuple[str, str], float],
        position_history: Sequence[tuple[float, float]],
        rsu_positions: Mapping[str, tuple[float, float]],
        uav_positions: Mapping[str, tuple[float, float]],
        uav_position_histories: Mapping[
            str, Sequence[tuple[float, float]]
        ] | None = None,
    ) -> dict[str, object]:
        causal_enabled = causal_contact_features_enabled(
            getattr(self.spec, "method", "UAMCO-DAG"),
            getattr(self.spec, "variant", None),
        )
        required_files = state.instance.required_input_files(candidate_task_id)
        horizon_s = float(self.config["timing"]["macro_interval_s"])
        sample_interval_s = float(self.config["timing"]["micro_slot_s"])
        maximum_link_rate = float(env.link_rate_bps)
        rates: dict[str, float] = {}
        queue_pressures: dict[str, float] = {}
        contact_margins: dict[str, float] = {}
        delivery_feasible: dict[str, float] = {}
        predicted_finish_times_s: dict[str, float] = {}
        remaining_budget_s = remaining_workflow_budget_s(
            current_time_s=float(env.current_time_s),
            workflow_deadline_s=float(state.deadline_time_s),
            episode_end_s=float(env.episode_s),
        )
        predictor = (
            CausalContactPredictor(
                history_length=min(5, len(position_history))
            )
            if causal_enabled and len(position_history) >= 2
            else None
        )
        infrastructure_positions = {**rsu_positions, **uav_positions}
        for action in ("local", *self.infrastructure_actions):
            target_executor = owner_ugv if action == "local" else action
            compute_queue = env.compute_queues.get(target_executor)
            running_item = env.running_compute.get(target_executor)
            cpu_hz = max(
                1.0,
                float(env.executor_cpu_hz.get(target_executor, 0.0)),
            )
            queued_cycles = sum(
                max(0.0, float(getattr(item, "remaining_cycles", 0.0)))
                for item in (
                    compute_queue.items() if compute_queue is not None else ()
                )
            )
            if running_item is not None:
                queued_cycles += max(
                    0.0,
                    float(getattr(running_item, "remaining_cycles", 0.0)),
                )
            task = state.instance.tasks[candidate_task_id]
            queue_pressures[action] = self._effective_queue_pressure(
                compute_queue,
                running_item=running_item,
                cpu_hz=cpu_hz,
                horizon_s=horizon_s,
            )
            missing_files = state.file_ledger.missing_at(
                required_files,
                target_executor,
            )
            hop_bytes: dict[str, int] = {}
            for file_id, size_bytes in missing_files:
                locations = state.file_ledger.locations(file_id)
                if not locations:
                    continue
                source = (
                    owner_ugv if owner_ugv in locations else locations[0]
                )
                route = env._transfer_route(
                    source,
                    owner_ugv,
                    target_executor,
                )
                for route_source, route_destination in zip(route, route[1:]):
                    infrastructure = (
                        route_destination
                        if route_source == owner_ugv
                        else route_source
                    )
                    hop_bytes[infrastructure] = (
                        hop_bytes.get(infrastructure, 0) + int(size_bytes)
                    )
            relevant_infrastructure = set(hop_bytes)
            if action != "local":
                relevant_infrastructure.add(action)
            route_rates: list[float] = []
            contact_windows: dict[str, float] = {}
            hop_rates: dict[str, float] = {}
            total_delivery_time_s = 0.0
            for infrastructure in relevant_infrastructure:
                rate = max(
                    0.0,
                    float(link_rates.get((owner_ugv, infrastructure), 0.0)),
                )
                hop_rates[infrastructure] = rate
                route_rates.append(rate)
                total_delivery_time_s += (
                    8.0 * hop_bytes.get(infrastructure, 0) / max(rate, 1.0)
                )
                position = infrastructure_positions.get(infrastructure)
                radius = float(
                    self.config["scenario"].get(
                        (
                            "rsu_radius_m"
                            if infrastructure.startswith("rsu-")
                            else "uav_radius_m"
                        ),
                        500.0 if infrastructure.startswith("rsu-") else 400.0,
                    )
                )
                effective_radius = radius * (
                    1.0 - float(getattr(self, "_contact_forecast_error", 0.0))
                )
                if predictor is None or position is None or rate <= 0.0:
                    contact_windows[infrastructure] = 0.0
                else:
                    contact_windows[infrastructure] = (
                        predictor.forecast_contact_window_s(
                            position_history,
                            infrastructure_xy=position,
                            radius_m=effective_radius,
                            horizon_s=horizon_s,
                            sample_interval_s=sample_interval_s,
                            infrastructure_history_xy=(
                                (uav_position_histories or {}).get(
                                    infrastructure
                                )
                            ),
                        )
                    )
            required_contact_s = {
                infrastructure: total_delivery_time_s
                for infrastructure in relevant_infrastructure
            }
            if action != "local":
                required_contact_s[action] = required_remote_contact_s(
                    is_sink=requires_causal_result_return(
                        is_sink=not bool(task.children),
                        feeds_high_fan_in_sink=any(
                            env._is_high_fan_in_sink(state, child_id)
                            for child_id in task.children
                        ),
                    ),
                    input_transfer_s=total_delivery_time_s,
                    queued_cycles=queued_cycles,
                    task_cycles=float(task.cycles),
                    cpu_hz=cpu_hz,
                    output_bytes=int(task.output_bytes),
                    link_rate_bps=hop_rates.get(action, 0.0),
                )
            requires_return = action != "local" and requires_causal_result_return(
                is_sink=not bool(task.children),
                feeds_high_fan_in_sink=any(
                    env._is_high_fan_in_sink(state, child_id)
                    for child_id in task.children
                ),
            )
            output_return_s = 0.0
            if requires_return and int(task.output_bytes) > 0:
                output_rate = max(0.0, float(hop_rates.get(action, 0.0)))
                output_return_s = (
                    math.inf
                    if output_rate <= 0.0
                    else 8.0 * int(task.output_bytes) / output_rate
                )
            predicted_finish_times_s[action] = predicted_executor_finish_time_s(
                input_transfer_s=total_delivery_time_s,
                queued_cycles=queued_cycles,
                task_cycles=float(task.cycles),
                cpu_hz=cpu_hz,
                output_return_s=output_return_s,
            )
            if not relevant_infrastructure:
                rates[action] = maximum_link_rate
                contact_margins[action] = float(causal_enabled)
                delivery_feasible[action] = float(causal_enabled)
            else:
                rates[action] = (
                    min(route_rates) if route_rates else 0.0
                )
                minimum_margin_s = min(
                    contact_windows.get(infrastructure, 0.0)
                    - required_contact_s[infrastructure]
                    for infrastructure in relevant_infrastructure
                )
                margin = (
                    minimum_margin_s
                ) / max(horizon_s, sample_interval_s)
                contact_margins[action] = float(
                    max(-2.0, min(1.0, margin))
                )
                delivery_feasible[action] = float(
                    causal_enabled
                    and target_executor in env.executor_cpu_hz
                    and bool(route_rates)
                    and all(rate > 0.0 for rate in route_rates)
                    and minimum_margin_s >= 0.0
                )
        return {
            "executor_link_rates_bps": rates,
            "executor_queue_pressures": queue_pressures,
            "executor_contact_margins": contact_margins,
            "executor_delivery_feasible": delivery_feasible,
            "enable_causal_contact_features": causal_enabled,
            "predicted_finish_times_s": predicted_finish_times_s,
            "remaining_budget_s": remaining_budget_s,
        }

    def _deadline_s(
        self,
        instance: WorkflowInstance,
        tier: str,
        *,
        experiment_mode: str,
    ) -> float:
        resources = self.config["resources"]
        bounds = workflow_completion_feasibility_s(
            instance,
            max_cpu_hz=max(float(value) for value in resources["executor_cpu_hz"].values()),
            fair_share_cpu_hz=workflow_accessible_fair_share_cpu_hz(
                self.config,
                experiment_mode=experiment_mode,
            ),
            max_link_rate_bps=float(resources["link_rate_bps"]),
            assignment_slot_s=float(self.config["timing"]["micro_slot_s"]),
        )
        multiplier = (
            float(self.config["sla"]["tiers"][tier]["deadline_multiplier"])
            * float(getattr(self, "_deadline_multiplier_scale", 1.0))
        )
        return max(
            float(self.config["timing"]["micro_slot_s"]),
            bounds["completion_feasibility_s"] * multiplier,
        )

    @staticmethod
    def _sample_tier(rng: random.Random, mix: Sequence[float]) -> str:
        draw = rng.random()
        cumulative = 0.0
        for tier, probability in zip(SLA_TIERS, mix):
            cumulative += float(probability)
            if draw <= cumulative:
                return tier
        return SLA_TIERS[-1]

    def _admit_one(
        self,
        env: UAMCOEnv,
        workflows: Sequence[WorkflowInstance],
        owner_ugv: str,
        *,
        rng: random.Random,
        experiment_mode: str,
    ) -> bool:
        if not workflows:
            raise ValueError("workflow split cannot be empty")
        instance = workflows[rng.randrange(len(workflows))]
        tier = self._sample_tier(rng, self.config["sla"]["mix"])
        try:
            env.admit_workflow(
                instance,
                owner_ugv=owner_ugv,
                sla_tier=tier,
                deadline_s=self._deadline_s(
                    instance,
                    tier,
                    experiment_mode=experiment_mode,
                ),
            )
        except OverflowError:
            return False
        return True

    def _sample_weight(self) -> float:
        """Return a fixed shared-interface scalar, not a policy preference."""
        return validate_interface_scalar(
            self.config.get("objective", {}).get("interface_scalar", 0.5),
            context=(
                f"job={getattr(self.spec, 'job_id', 'unknown')} "
                f"method={getattr(self.spec, 'method', 'unknown')} "
                f"fold={getattr(self.spec, 'fold', 'unknown')} "
                f"seed={getattr(self.spec, 'seed', 'unknown')} training"
            ),
        )

    @staticmethod
    def _executor_queue_work(env: UAMCOEnv, executor_id: str) -> float:
        work = 0.0
        queue = env.compute_queues.get(executor_id)
        if queue is not None:
            work += sum(float(item.remaining_cycles) for item in queue.items())
        running = env.running_compute.get(executor_id)
        if running is not None:
            work += float(running.remaining_cycles)
        return work

    def _build_sota_observation(
        self,
        *,
        env: UAMCOEnv,
        workflow_id: str,
        candidate_task_id: str,
        graph_observation: GraphObservation,
        link_rates: Mapping[tuple[str, str], float],
    ) -> SOTAObservation:
        state = env.workflow_states[workflow_id]
        task = state.instance.tasks[candidate_task_id]
        task_index = graph_observation.task_ids.index(candidate_task_id)
        action_mask = tuple(
            bool(value)
            for value in graph_observation.action_mask[0, task_index].tolist()
        )
        actual_executors = tuple(
            state.owner_ugv if action == "local" else action
            for action in graph_observation.executor_actions
        )
        queue_work: list[float] = []
        capacities: list[float] = []
        rates: list[float] = []
        remaining_energy: list[float] = []
        for action, actual in zip(graph_observation.executor_actions, actual_executors):
            if action == "defer":
                queue_work.append(0.0)
                capacities.append(0.0)
                rates.append(0.0)
                remaining_energy.append(math.inf)
            else:
                queue_work.append(self._executor_queue_work(env, actual))
                capacities.append(float(env.executor_cpu_hz.get(actual, 0.0)))
                rates.append(
                    math.inf
                    if action == "local"
                    else float(link_rates.get((state.owner_ugv, action), 0.0))
                )
                remaining_energy.append(env.remaining_energy_j(actual))
        return SOTAObservation(
            candidate_task_id=f"{workflow_id}/{candidate_task_id}",
            owner_index=int(state.owner_ugv.split("-")[-1]),
            input_bytes=float(task.input_bytes),
            cycles=float(task.cycles),
            executor_ids=tuple(graph_observation.executor_actions),
            executor_queue_work=tuple(queue_work),
            executor_cpu_hz=tuple(capacities),
            link_rate_bps=tuple(rates),
            action_mask=action_mask,
            remaining_energy_j=tuple(remaining_energy),
            micro_slot=int(env.micro_slot_index),
        )

    def _build_absorbing_sota_observation(
        self,
        *,
        env: UAMCOEnv,
        owner_ugv: str,
        previous: SOTAObservation,
        link_rates: Mapping[tuple[str, str], float],
    ) -> SOTAObservation:
        """Build a real post-step state when this owner has no next decision."""
        queue_work: list[float] = []
        capacities: list[float] = []
        rates: list[float] = []
        remaining_energy: list[float] = []
        for action in previous.executor_ids:
            actual = owner_ugv if action == "local" else action
            if action == "defer":
                queue_work.append(0.0); capacities.append(0.0)
                rates.append(0.0); remaining_energy.append(math.inf)
            else:
                queue_work.append(self._executor_queue_work(env, actual))
                capacities.append(float(env.executor_cpu_hz.get(actual, 0.0)))
                rates.append(math.inf if action == "local" else float(link_rates.get((owner_ugv, action), 0.0)))
                remaining_energy.append(env.remaining_energy_j(actual))
        defer_index = previous.executor_ids.index("defer")
        mask = tuple(index == defer_index for index in range(len(previous.executor_ids)))
        return SOTAObservation(
            candidate_task_id=f"{owner_ugv}/absorbing",
            owner_index=previous.owner_index,
            input_bytes=0.0,
            cycles=0.0,
            executor_ids=previous.executor_ids,
            executor_queue_work=tuple(queue_work),
            executor_cpu_hz=tuple(capacities),
            link_rate_bps=tuple(rates),
            action_mask=mask,
            remaining_energy_j=tuple(remaining_energy),
            micro_slot=int(env.micro_slot_index),
            absorbing=True,
        )

    @staticmethod
    def _sota_physical_context(
        *,
        config: Mapping,
        env: UAMCOEnv,
        placements: Mapping[str, TrajectoryPlacement],
        ugv_positions: Mapping[str, tuple[float, float]],
        uav_positions: Mapping[str, tuple[float, float]],
    ) -> SOTAPhysicalContext:
        owner_ids = tuple(placements)
        active_owners = {
            state.owner_ugv
            for state in env.workflow_states.values()
            if state.status == "active"
        }
        return SOTAPhysicalContext(
            bandwidth_hz=float(config["resources"]["channel"]["bandwidth_hz"]),
            owner_positions_xy_m=tuple(ugv_positions[owner] for owner in owner_ids),
            unfinished_owner_mask=tuple(owner in active_owners for owner in owner_ids),
            uav_ids=tuple(uav_positions),
        )

    def _run_episode(
        self,
        *,
        workflows: Sequence[WorkflowInstance],
        traces: Sequence[MobilityTrace],
        delay_weight: float,
        training: bool,
        deterministic: bool,
        episode_seed: int,
        experiment_mode: str | None = None,
        workflow_feasibility_config: Mapping | None = None,
        progress_episode: int | None = None,
    ) -> EpisodeRollout:
        delay_weight = validate_interface_scalar(
            delay_weight,
            context=(
                f"job={getattr(self.spec, 'job_id', 'unknown')} "
                f"method={getattr(self.spec, 'method', 'unknown')} "
                f"fold={getattr(self.spec, 'fold', 'unknown')} "
                f"seed={getattr(self.spec, 'seed', 'unknown')} "
                f"episode_seed={episode_seed} training={training}"
            ),
        )
        rng = random.Random(int(episode_seed))
        resolved_experiment_mode = str(
            experiment_mode
            or self.config.get("training", {}).get(
                "experiment_mode", "main_completion"
            )
        )
        episode_workflows = select_workflows_for_experiment(
            workflows,
            config=self.config,
            feasibility_config=workflow_feasibility_config,
            experiment_mode=resolved_experiment_mode,
        )
        if training and len(episode_workflows) >= 3:
            strata = stratify_workflows_by_work(episode_workflows)
            episode_index = int(episode_seed) % 100_000
            curriculum = self.config.get("training", {}).get("workload_curriculum", {})
            stage = curriculum_stage(
                completed_episode=episode_index,
                total_episodes=int(self.spec.episodes),
                curriculum=curriculum,
            )
            if stage == "easy":
                episode_workflows = strata[0]
            elif stage == "medium":
                episode_workflows = strata[0] + strata[1]
            else:
                episode_workflows = tuple(
                    instance for stratum in strata for instance in stratum
                )
        episode_started = time.monotonic()
        progress_interval_started = episode_started
        progress_interval_slots = int(
            self.config.get("training", {}).get("progress_micro_slot_interval", 100)
        )
        if progress_interval_slots <= 0:
            raise ValueError("micro-slot progress interval must be positive")
        episode_decision_count = 0
        inference_wall_s = 0.0
        finish_time_action_correction_count = 0
        maximum_predicted_finish_time_s = 0.0
        uav_endpoint_refinement_assignment_count = 0
        maximum_executor_backlog_s = 0.0
        env = self._new_env(experiment_mode=resolved_experiment_mode)
        allow_dynamic_admissions = not bool(
            self.config.get("experiments", {})
            .get(resolved_experiment_mode, {})
            .get("stop_new_admissions", False)
        )
        if self.sota_runtime is not None:
            self.sota_runtime.reset_episode(int(episode_seed))
        placements = self._trajectory_placements(traces, rng=rng)
        rsu_positions = self._rsu_positions()
        uav_positions = self._initial_uav_positions()
        uav_position_histories = {
            uav_id: [position]
            for uav_id, position in uav_positions.items()
        }
        uav_targets = dict(uav_positions)
        previous_ugv_positions = {
            ugv_id: placement.position_at(0.0) for ugv_id, placement in placements.items()
        }
        position_histories = {
            ugv_id: [position] for ugv_id, position in previous_ugv_positions.items()
        }
        distance_since_arrival = {ugv_id: 0.0 for ugv_id in placements}
        admitted_count = 0
        admission_rejections = 0
        initial_owners = list(placements)
        rng.shuffle(initial_owners)
        active_workflow_limit = int(
            self.config.get("experiments", {})
            .get(resolved_experiment_mode, {})
            .get(
                "active_workflows_per_episode",
                self.config["scenario"].get(
                    "active_workflows_per_episode",
                    4,
                ),
            )
        )
        initial_instances = (
            sample_stratified_workflows(
                episode_workflows,
                count=active_workflow_limit,
                rng=rng,
            )
            if episode_workflows
            else (None,) * active_workflow_limit
        )
        for ugv_id, initial_instance in zip(
            initial_owners[:active_workflow_limit],
            initial_instances,
        ):
            admitted_count += int(
                self._admit_one(
                    env,
                    (
                        (initial_instance,)
                        if initial_instance is not None
                        else episode_workflows
                    ),
                    ugv_id,
                    rng=rng,
                    experiment_mode=resolved_experiment_mode,
                )
            )
        if env.termination_mode == "drain_admitted_workflows":
            env.close_admissions()
        all_decisions: list[RolloutDecision] = []
        dt_s = float(self.config["timing"]["micro_slot_s"])
        macro_steps = max(1, int(round(float(self.config["timing"]["macro_interval_s"]) / dt_s)))
        trigger_distance = float(self.config["scenario"]["workflow_trigger_distance_m"])
        if admitted_count <= 0:
            raise RuntimeError("episode admitted no workflows; objective is undefined")
        objective = FiniteHorizonObjectiveAccumulator(
            self.objective_bounds, admitted_count=admitted_count
        )
        initial_progress_potential = actual_delivery_progress_potential(env)
        previous_progress_potential = initial_progress_potential
        pending_decisions: list[RolloutDecision] = []
        pending_reward = 0.0
        incremental_episode_return = 0.0
        pending_costs = {key: 0.0 for key in self.cost_keys}
        sota_update_records: list[Mapping[str, float]] = []
        terminal_sota_actions: list[
            tuple[SOTAObservation, SOTAAction, float, SOTAObservation]
        ] = []
        rsu_connected_pairs = 0
        rsu_observed_pairs = 0

        def settle_pending() -> None:
            nonlocal pending_decisions, pending_reward, pending_costs
            if pending_decisions:
                reward_share = pending_reward / len(pending_decisions)
                for decision in pending_decisions:
                    decision.reward = reward_share
                    decision.costs = {
                        key: pending_costs[key] / len(pending_decisions)
                        for key in self.cost_keys
                    }
                all_decisions.extend(pending_decisions)
            pending_decisions = []
            pending_reward = 0.0
            pending_costs = {key: 0.0 for key in self.cost_keys}
        step_index = 0
        while True:
            ugv_positions = {
                ugv_id: placement.position_at(env.current_time_s)
                for ugv_id, placement in placements.items()
            }
            for ugv_id, position in ugv_positions.items():
                distance_since_arrival[ugv_id] += math.dist(previous_ugv_positions[ugv_id], position)
                while (
                    env.termination_mode == "fixed_horizon"
                    and allow_dynamic_admissions
                    and distance_since_arrival[ugv_id] + 1e-12 >= trigger_distance
                ):
                    distance_since_arrival[ugv_id] -= trigger_distance
                    active_count = sum(
                        state.status == "active"
                        for state in env.workflow_states.values()
                    )
                    if active_count >= active_workflow_limit:
                        continue
                    if self._admit_one(
                        env,
                        episode_workflows,
                        ugv_id,
                        rng=rng,
                        experiment_mode=resolved_experiment_mode,
                    ):
                        admitted_count += 1
                        objective.set_admitted_count(admitted_count)
                    else:
                        admission_rejections += 1
                previous_ugv_positions[ugv_id] = position
                position_histories[ugv_id].append(position)

            current_zone_demand, current_zone_targets = zone_delivery_profile(
                env.workflow_states,
                ugv_positions,
                width_m=float(self.config["scenario"].get("width_m", 3000.0)),
                height_m=float(self.config["scenario"].get("height_m", 3000.0)),
            )
            connectivity = self._connectivity(ugv_positions, rsu_positions, uav_positions)
            rsu_links = tuple(
                connected
                for (_, infrastructure_id), connected in connectivity.items()
                if infrastructure_id.startswith("rsu-")
            )
            rsu_connected_pairs += sum(int(connected) for connected in rsu_links)
            rsu_observed_pairs += len(rsu_links)
            link_rates = self._link_rates(
                ugv_positions,
                rsu_positions,
                uav_positions,
                connectivity,
            )
            step_decisions: list[RolloutDecision] = []
            step_sota_actions: list[tuple[str, SOTAObservation, SOTAAction]] = []
            step_neural_requests: list[
                tuple[str, str, str, object, GraphObservation, Mapping[str, object]]
            ] = []
            assignments: dict[tuple[str, str], str] = {}
            macro_due = (
                step_index % macro_steps == 0
                and not (
                    delivery_guard_enabled(
                        getattr(self.spec, "method", "UAMCO-DAG"),
                        getattr(self.spec, "variant", None),
                    )
                    and env.causal_uav_delivery_guard_active()
                )
            )
            macro_selected = False
            for owner_ugv in placements:
                candidate = select_fixed_candidate(env, owner_ugv)
                if candidate is None:
                    continue
                workflow_id = candidate.workflow_id
                state = env.workflow_states[workflow_id]
                reachable = tuple(
                    action
                    for action in self.infrastructure_actions
                    if connectivity.get((owner_ugv, action), False)
                )
                causal_enabled = causal_contact_features_enabled(
                    getattr(self.spec, "method", "UAMCO-DAG"),
                    getattr(self.spec, "variant", None),
                )
                executor_context = self._executor_observation_context(
                    env=env,
                    state=state,
                    candidate_task_id=candidate.task_id,
                    owner_ugv=owner_ugv,
                    link_rates=link_rates,
                    position_history=position_histories[state.owner_ugv],
                    rsu_positions=rsu_positions,
                    uav_positions=uav_positions,
                    uav_position_histories=uav_position_histories,
                )
                observation = build_graph_observation(
                    env,
                    workflow_id,
                    candidate_task_id=candidate.task_id,
                    infrastructure_actions=self.infrastructure_actions,
                    current_reachable_executors=reachable,
                    delay_weight=delay_weight,
                    contact_history_score=(
                        self._contact_history_score(
                            position_histories[state.owner_ugv],
                            rsu_positions,
                            uav_positions,
                            uav_position_histories,
                        )
                        if causal_enabled
                        else 0.0
                    ),
                    max_nodes=int(
                        self.config.get("algorithm", {}).get(
                            "max_observation_nodes", 64
                        )
                    ),
                    zone_delivery_demand=(
                        current_zone_demand
                        if causal_enabled
                        else (0.0,) * 9
                    ),
                    **graph_observation_executor_context(executor_context),
                )
                if self.is_domain_sota:
                    if self.sota_runtime is None:
                        raise RuntimeError("domain SOTA route has no method runtime")
                    sota_observation = self._build_sota_observation(
                        env=env,
                        workflow_id=workflow_id,
                        candidate_task_id=candidate.task_id,
                        graph_observation=observation,
                        link_rates=link_rates,
                    )
                    physical_context = self._sota_physical_context(
                        config=self.config,
                        env=env,
                        placements=placements,
                        ugv_positions=ugv_positions,
                        uav_positions=uav_positions,
                    )
                    inference_started = time.monotonic()
                    sota_action = self.sota_runtime.act(
                        sota_observation,
                        physical_context,
                        deterministic=deterministic,
                    )
                    inference_wall_s += max(
                        0.0, time.monotonic() - inference_started
                    )
                    action_name = observation.executor_actions[
                        sota_action.executor_index
                    ]
                    if action_name == "local":
                        action_name = state.owner_ugv
                    assignments[(workflow_id, candidate.task_id)] = action_name
                    step_sota_actions.append((owner_ugv, sota_observation, sota_action))
                    if macro_due and not macro_selected and sota_action.macro_targets_xy_m:
                        uav_ids = tuple(uav_positions)
                        if len(sota_action.macro_targets_xy_m) != len(uav_ids):
                            raise ValueError("domain SOTA macro target count differs from active UAVs")
                        uav_targets = dict(zip(uav_ids, sota_action.macro_targets_xy_m))
                        env.step_macro(uav_targets)
                        macro_selected = True
                    continue
                step_neural_requests.append(
                    (
                        owner_ugv,
                        workflow_id,
                        candidate.task_id,
                        state,
                        observation,
                        executor_context,
                    )
                )

            if step_neural_requests:
                if self.policy is None:
                    raise RuntimeError("PPO/MARL route has no policy")
                inference_started = time.monotonic()
                neural_results = sample_policy_observation_batch(
                    self.policy,
                    tuple(request[4] for request in step_neural_requests),
                    device=self.device,
                    deterministic=deterministic,
                )
                inference_wall_s += max(
                    0.0, time.monotonic() - inference_started
                )
                for request_index, (request, policy_result) in enumerate(
                    zip(step_neural_requests, neural_results)
                ):
                    (
                        _,
                        workflow_id,
                        candidate_task_id,
                        state,
                        observation,
                        executor_context,
                    ) = request
                    macro_action = None
                    macro_log_probability = 0.0
                    if (
                        request_index == 0
                        and macro_due
                        and not macro_selected
                        and self.spec.variant != "non_hierarchical"
                    ):
                        macro_distribution = Categorical(logits=policy_result.macro_logits)
                        sampled_macro = (
                            policy_result.macro_logits.argmax(dim=-1)
                            if deterministic
                            else macro_distribution.sample()
                        )
                        macro_action = int(sampled_macro.item())
                        macro_log_probability = float(
                            macro_distribution.log_prob(sampled_macro).item()
                        )
                        uav_targets = self._macro_targets(
                            macro_action,
                            zone_targets=current_zone_targets,
                        )
                        env.step_macro(uav_targets)
                        macro_selected = True
                    node_index = observation.task_ids.index(candidate_task_id)
                    policy_action_name = observation.executor_actions[
                        int(policy_result.actions[0, node_index])
                    ]
                    predicted_times = {
                        str(action): float(value)
                        for action, value in dict(
                            executor_context["predicted_finish_times_s"]
                        ).items()
                    }
                    finite_predictions = tuple(
                        value
                        for value in predicted_times.values()
                        if math.isfinite(value)
                    )
                    if finite_predictions:
                        maximum_predicted_finish_time_s = max(
                            maximum_predicted_finish_time_s,
                            max(finite_predictions),
                        )
                    action_name = policy_action_name
                    if finish_time_guard_enabled(
                        getattr(self.spec, "method", "UAMCO-DAG"),
                        getattr(self.spec, "variant", None),
                    ):
                        legal_actions = tuple(
                            action
                            for action_index, action in enumerate(
                                observation.executor_actions
                            )
                            if bool(
                                observation.action_mask[
                                    0, node_index, action_index
                                ].item()
                            )
                        )
                        action_name = guard_executor_action(
                            selected_action=policy_action_name,
                            legal_actions=legal_actions,
                            predicted_finish_times_s=predicted_times,
                            remaining_budget_s=float(
                                executor_context["remaining_budget_s"]
                            ),
                            macro_interval_s=float(
                                self.config["timing"]["macro_interval_s"]
                            ),
                        )
                        if action_name != policy_action_name:
                            finish_time_action_correction_count += 1
                    if action_name == "local":
                        action_name = state.owner_ugv
                    assignments[(workflow_id, candidate_task_id)] = action_name
                    if training:
                        step_decisions.append(
                            RolloutDecision(
                                observation=observation,
                                actions=policy_result.actions,
                                old_log_probability=(
                                    policy_result.old_log_probability
                                    + macro_log_probability
                                ),
                                reward_value=policy_result.reward_value,
                                cost_values=policy_result.cost_values,
                                macro_action=macro_action,
                                macro_old_log_probability=macro_log_probability,
                                costs={key: 0.0 for key in self.cost_keys},
                            )
                        )

            if file_endpoint_refinement_enabled(
                getattr(self.spec, "method", "UAMCO-DAG"),
                getattr(self.spec, "variant", None),
            ):
                refined_targets = causal_uav_service_targets(
                    env,
                    ugv_positions,
                    uav_ids=tuple(uav_positions),
                    current_time_s=float(env.current_time_s),
                    episode_end_s=float(env.episode_s),
                    enable_file_endpoint=True,
                    enable_idle_deficit=idle_uav_deficit_support_enabled(
                        getattr(self.spec, "method", "UAMCO-DAG"),
                        getattr(self.spec, "variant", None),
                    )
                    and idle_uav_deficit_update_due(
                        step_index=step_index,
                        macro_steps=macro_steps,
                    ),
                )
                uav_targets.update(refined_targets)
                uav_endpoint_refinement_assignment_count += len(refined_targets)
            uav_positions, uav_speeds = self._move_uavs(uav_positions, uav_targets)
            for uav_id, position in uav_positions.items():
                uav_position_histories[uav_id].append(position)
            if step_decisions:
                settle_pending()
                pending_decisions = step_decisions
            active_before_step = tuple(
                workflow_id
                for workflow_id, state in env.workflow_states.items()
                if state.status == "active"
            )
            energy_before = env.mobile_energy_j + env.rsu_energy_j
            env.record_uav_flight_energy(uav_speeds, dt_s=dt_s)
            step_result = env.step_micro(
                assignments,
                current_connectivity=connectivity,
                link_rates_bps=link_rates,
            )
            backlog_by_executor = executor_backlog_seconds(env)
            if backlog_by_executor:
                maximum_executor_backlog_s = max(
                    maximum_executor_backlog_s,
                    max(backlog_by_executor.values()),
                )
            energy_increment = env.mobile_energy_j + env.rsu_energy_j - energy_before
            increment = objective.advance(
                active_workflow_ids=active_before_step,
                dt_s=dt_s,
                system_energy_increment_j=energy_increment,
            )
            current_progress_potential = actual_delivery_progress_potential(env)
            progress_reward = (
                current_progress_potential - previous_progress_potential
            )
            previous_progress_potential = current_progress_potential
            step_reward = -increment["edp"] + progress_reward
            pending_reward += step_reward
            incremental_episode_return += step_reward
            for key in self.cost_keys:
                pending_costs[key] += float(step_result.costs[key])
            if training and step_sota_actions:
                next_ugv_positions = {
                    ugv_id: placement.position_at(env.current_time_s)
                    for ugv_id, placement in placements.items()
                }
                next_connectivity = self._connectivity(
                    next_ugv_positions, rsu_positions, uav_positions
                )
                next_link_rates = self._link_rates(
                    next_ugv_positions,
                    rsu_positions,
                    uav_positions,
                    next_connectivity,
                )
                reward_share = step_reward / len(step_sota_actions)
                for owner_ugv, sota_observation, sota_action in step_sota_actions:
                    next_candidate = None if step_result.terminated else select_fixed_candidate(env, owner_ugv)
                    if next_candidate is None:
                        next_observation = self._build_absorbing_sota_observation(
                            env=env,
                            owner_ugv=owner_ugv,
                            previous=sota_observation,
                            link_rates=next_link_rates,
                        )
                    else:
                        next_state = env.workflow_states[next_candidate.workflow_id]
                        next_reachable = tuple(
                            action for action in self.infrastructure_actions
                            if next_connectivity.get((owner_ugv, action), False)
                        )
                        next_history = tuple(
                            position_histories[next_state.owner_ugv]
                        )
                        next_owner_position = next_ugv_positions[
                            next_state.owner_ugv
                        ]
                        if (
                            not next_history
                            or next_history[-1] != next_owner_position
                        ):
                            next_history = (*next_history, next_owner_position)
                        next_causal_enabled = causal_contact_features_enabled(
                            getattr(self.spec, "method", "UAMCO-DAG"),
                            getattr(self.spec, "variant", None),
                        )
                        next_executor_context = (
                            self._executor_observation_context(
                                env=env,
                                state=next_state,
                                candidate_task_id=next_candidate.task_id,
                                owner_ugv=next_state.owner_ugv,
                                link_rates=next_link_rates,
                                position_history=next_history,
                                rsu_positions=rsu_positions,
                                uav_positions=uav_positions,
                                uav_position_histories=uav_position_histories,
                            )
                        )
                        next_graph_observation = build_graph_observation(
                            env,
                            next_candidate.workflow_id,
                            candidate_task_id=next_candidate.task_id,
                            infrastructure_actions=self.infrastructure_actions,
                            current_reachable_executors=next_reachable,
                            delay_weight=delay_weight,
                            contact_history_score=(
                                self._contact_history_score(
                                    next_history,
                                    rsu_positions,
                                    uav_positions,
                                    uav_position_histories,
                                )
                                if next_causal_enabled
                                else 0.0
                            ),
                            max_nodes=int(
                                self.config.get("algorithm", {}).get(
                                    "max_observation_nodes", 64
                                )
                            ),
                            zone_delivery_demand=(
                                zone_delivery_demand(
                                    env.workflow_states,
                                    next_ugv_positions,
                                    width_m=float(
                                        self.config["scenario"].get(
                                            "width_m",
                                            3000.0,
                                        )
                                    ),
                                    height_m=float(
                                        self.config["scenario"].get(
                                            "height_m",
                                            3000.0,
                                        )
                                    ),
                                )
                                if next_causal_enabled
                                else (0.0,) * 9
                            ),
                            **graph_observation_executor_context(
                                next_executor_context
                            ),
                        )
                        next_observation = self._build_sota_observation(
                            env=env,
                            workflow_id=next_candidate.workflow_id,
                            candidate_task_id=next_candidate.task_id,
                            graph_observation=next_graph_observation,
                            link_rates=next_link_rates,
                        )
                    transition_done = bool(step_result.terminated or next_observation.absorbing)
                    if step_result.terminated:
                        terminal_sota_actions.append(
                            (sota_observation, sota_action, reward_share, next_observation)
                        )
                    else:
                        self.sota_runtime.observe(
                            SOTATransition(
                                observation=sota_observation,
                                action=sota_action,
                                reward=reward_share,
                                next_observation=next_observation,
                                done=transition_done,
                            )
                        )
                if not step_result.terminated and getattr(self.spec, "method", "") != "MEC-UARA":
                    sota_update_records.append(self.sota_runtime.update_if_ready())
            if training and getattr(self.spec, "method", "") == "MEC-UARA":
                sota_update_records.append(
                    self.sota_runtime.tick_micro_slot(int(env.micro_slot_index))
                )
            episode_decision_count += len(step_neural_requests) + len(step_sota_actions)
            step_index += 1
            if (
                training
                and progress_episode is not None
                and env.micro_slot_index % progress_interval_slots == 0
            ):
                self._write_micro_slot_progress(
                    episode=progress_episode,
                    env=env,
                    decision_count=episode_decision_count,
                    episode_started=episode_started,
                    interval_started=progress_interval_started,
                    interval_slots=progress_interval_slots,
                )
                progress_interval_started = time.monotonic()
            if step_result.terminated:
                break

        per_workflow_mobile = env.mobile_energy_j / max(1, admitted_count)
        per_workflow_rsu = env.rsu_energy_j / max(1, admitted_count)
        episode_duration_s = (
            env.episode_s
            if env.termination_mode == "fixed_horizon"
            else env.current_time_s
        )
        outcomes: list[WorkflowOutcome] = []
        for workflow_id, state in env.workflow_states.items():
            status = state.status if state.status in {"completed", "dropped"} else "remaining"
            outcomes.append(
                WorkflowOutcome(
                    workflow_id=workflow_id,
                    family=state.instance.family,
                    sla_tier=state.sla_tier,
                    arrival_time_s=state.arrival_time_s,
                    deadline_time_s=state.deadline_time_s,
                    censor_time_s=episode_duration_s,
                    status=status,
                    completion_time_s=state.completion_time_s,
                    missed_deadline=state.missed_deadline,
                    mobile_energy_j=per_workflow_mobile,
                    rsu_energy_j=per_workflow_rsu,
                    drop_reason=state.drop_reason,
                )
            )
        metrics = compute_episode_metrics(
            outcomes,
            episode_duration_s=episode_duration_s,
            objective_bounds=self.objective_bounds,
        )
        terminal_correction = objective.finalize(
            outcomes, episode_duration_s=episode_duration_s
        )
        if not math.isclose(
            objective.delay_cost_total,
            float(metrics["normalized_effective_delay"]),
            abs_tol=1.0e-12,
        ) or not math.isclose(
            objective.energy_cost_total,
            float(metrics["normalized_system_energy"]),
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("frozen objective accounting does not telescope to metrics")
        final_progress_potential = actual_delivery_progress_potential(env)
        failed_count = sum(outcome.status != "completed" for outcome in outcomes)
        dominance_coefficient = completion_dominance_coefficient(admitted_count)
        terminal_progress_correction = -(1.0 - initial_progress_potential)
        terminal_failure_cost = dominance_coefficient * failed_count / admitted_count
        terminal_reward = (
            -terminal_correction["edp"]
            + terminal_progress_correction
            - terminal_failure_cost
        )
        pending_reward += terminal_reward
        incremental_episode_return += terminal_reward
        if training and terminal_sota_actions:
            terminal_share = terminal_reward / len(terminal_sota_actions)
            for sota_observation, sota_action, step_share, next_observation in terminal_sota_actions:
                self.sota_runtime.observe(
                    SOTATransition(
                        observation=sota_observation,
                        action=sota_action,
                        reward=step_share + terminal_share,
                        next_observation=next_observation,
                        done=True,
                    )
                )
            if getattr(self.spec, "method", "") != "MEC-UARA":
                sota_update_records.append(self.sota_runtime.update_if_ready())
        expected_episode_return = finite_horizon_episode_return(
            normalized_delay=float(metrics["normalized_effective_delay"]),
            normalized_energy=float(metrics["normalized_system_energy"]),
            actual_delivery_progress=final_progress_potential,
            failed_count=failed_count,
            admitted_count=admitted_count,
        )
        if not math.isclose(
            incremental_episode_return,
            expected_episode_return,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                "reward increments do not telescope to the finite-horizon objective"
            )
        settle_pending()
        if training and not self.is_domain_sota and not math.isclose(
            sum(decision.reward for decision in all_decisions),
            expected_episode_return,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError("training decision rewards do not sum to episode return")
        metrics.update(env.correctness_metrics())
        metrics.update(env.mechanism_metrics())
        total_tasks = sum(
            len(state.instance.tasks) for state in env.workflow_states.values()
        )
        completed_tasks = sum(
            len(state.compute_completed) for state in env.workflow_states.values()
        )
        metrics["task_compute_completion_ratio"] = (
            completed_tasks / max(1, total_tasks)
        )
        metrics["actual_delivery_progress_potential"] = final_progress_potential
        metrics["terminal_delivery_residual_cost"] = 1.0 - final_progress_potential
        metrics["failed_workflow_ratio"] = failed_count / admitted_count
        metrics["completion_dominance_coefficient"] = dominance_coefficient
        metrics["normalized_edp"] = bounded_unit(
            float(metrics["normalized_effective_delay"])
        ) * bounded_unit(float(metrics["normalized_system_energy"]))
        metrics["delivery_first_objective"] = -expected_episode_return
        metrics["policy_inference_wall_s"] = float(inference_wall_s)
        metrics["finish_time_action_correction_count"] = float(
            finish_time_action_correction_count
        )
        metrics["maximum_predicted_finish_time_s"] = float(
            maximum_predicted_finish_time_s
        )
        metrics["uav_endpoint_refinement_assignment_count"] = float(
            uav_endpoint_refinement_assignment_count
        )
        metrics["maximum_executor_backlog_s"] = float(
            maximum_executor_backlog_s
        )
        scheduled_by_executor: dict[str, int] = {}
        for state in env.workflow_states.values():
            for executor in state.scheduled.values():
                scheduled_by_executor[executor] = (
                    scheduled_by_executor.get(executor, 0) + 1
                )
        for executor in sorted(env.executor_cpu_hz):
            metrics[f"scheduled_task_count__{executor}"] = float(
                scheduled_by_executor.get(executor, 0)
            )
        metrics["simulated_episode_s"] = float(episode_duration_s)
        metrics["admission_rejection_count"] = float(admission_rejections)
        metrics["uav_flight_energy_j"] = float(env.uav_flight_energy_j)
        metrics["communication_energy_j"] = float(env.communication_energy_j)
        metrics["rsu_contact_ratio"] = (
            rsu_connected_pairs / max(1, rsu_observed_pairs)
        )
        for component, energy_j in env.energy_by_component_j.items():
            key = "energy_" + component.lower().replace(" ", "_") + "_per_admitted_dag_j"
            metrics[key] = float(energy_j) / max(1, admitted_count)
        return EpisodeRollout(
            decisions=tuple(all_decisions),
            outcomes=tuple(outcomes),
            metrics=metrics,
            admitted_count=admitted_count,
            episode_return=expected_episode_return,
            update_summary=(
                {
                    key: float(
                        mean(
                            float(record[key])
                            for record in sota_update_records
                            if key in record
                        )
                    )
                    for key in sorted(
                        {
                            key
                            for record in sota_update_records
                            for key in record
                        }
                    )
                }
                if sota_update_records
                else None
            ),
        )

    def run(self) -> None:
        is_smoke = self.spec.protocol == "smoke"
        run_started = time.monotonic()
        training_workflows = self.fold.train
        training_experiment_mode: str | None = None
        if is_smoke:
            training_workflows = resolve_smoke_workflows(
                self.fold,
                self.config,
            )
            training_experiment_mode = resolve_smoke_experiment_mode(
                self.config
            )
        start_episode, history = self._load_checkpoint()
        self._resume_completed_wall_time_s = sum(
            float(record.get("episode_wall_time_s", 0.0))
            for record in history
        )
        validation_interval = int(
            self.config["training"].get(
                "validation_interval",
                self.config["training"]["checkpoint_interval"],
            )
        )
        if (
            not is_smoke
            and start_episode > 0
            and start_episode % validation_interval == 0
            and history
            and "validation" not in history[-1]
        ):
            self._complete_validation(start_episode, history)
            self._save_checkpoint(start_episode, history)
            self._write_episode_progress(
                start_episode,
                history[-1],
                run_started=run_started,
            )
        for episode in range(start_episode, int(self.spec.episodes)):
            episode_wall_started = time.monotonic()
            delay_weight = self._sample_weight()
            rollout = self._run_episode(
                workflows=training_workflows,
                traces=self.mobility["rellis"]["train"],
                delay_weight=delay_weight,
                training=True,
                deterministic=False,
                episode_seed=self.spec.seed * 100_000 + episode,
                experiment_mode=training_experiment_mode,
                progress_episode=episode + 1,
            )
            update_started = time.monotonic()
            loss_summary = (
                dict(rollout.update_summary or {"updated": 0.0})
                if self.is_domain_sota
                else self._update_policy(rollout.decisions, rollout.admitted_count)
            )
            update_wall_s = max(0.0, time.monotonic() - update_started)
            history.append(
                {
                    "episode": episode + 1,
                    "delay_weight": delay_weight,
                    "admitted_workflows": rollout.admitted_count,
                    "episode_return": rollout.episode_return,
                    "metrics": rollout.metrics,
                    "loss": loss_summary,
                    "lagrange": self.lagrange.values(),
                    "episode_wall_time_s": max(
                        0.0, time.monotonic() - episode_wall_started
                    ),
                    "training_update_wall_s": update_wall_s,
                }
            )
            completed_episode = episode + 1
            if (
                not is_smoke
                and completed_episode
                % int(self.config["training"]["checkpoint_interval"])
                == 0
            ):
                self._save_checkpoint(completed_episode, history)
            self._write_episode_progress(
                completed_episode,
                history[-1],
                run_started=run_started,
            )
            if (
                not is_smoke
                and completed_episode % validation_interval == 0
            ):
                self._complete_validation(completed_episode, history)
                self._save_checkpoint(completed_episode, history)
                self._write_episode_progress(
                    completed_episode,
                    history[-1],
                    run_started=run_started,
                )
        if is_smoke:
            self._save_checkpoint(int(self.spec.episodes), history)
            evaluations = self._evaluate_smoke()
            sensitivity: list[dict] = []
            aggregate_metrics = self._aggregate_evaluations(evaluations)
        else:
            self._load_best_policy()
            evaluations = self._evaluate()
            sensitivity = self._evaluate_sensitivity()
            aggregate_metrics = self._aggregate_evaluations(evaluations)
        result = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state": "succeeded",
            "job": asdict(self.spec),
            "continuous_preference_training": False,
            "objective": "delivery_first_edp",
            "training_history": history,
            "evaluation": evaluations,
            "sensitivity": sensitivity,
            "provenance": {
                "workflow_source": "WfInstances production execution records",
                "workflow_generator_used": False,
                "mobility_training": "RELLIS-3D train split",
                "mobility_zero_shot": "M2DGR Outdoor",
                "mobility_zero_shot_eligible_sequences": [
                    trace.trace_id
                    for trace in (
                        getattr(self, "_m2dgr_selection", None).eligible
                        if getattr(self, "_m2dgr_selection", None) is not None
                        else ()
                    )
                ],
                "mobility_zero_shot_excluded_sequences": [
                    {
                        "trace_id": trace_id,
                        "duration_s": duration_s,
                        "reason": "shorter_than_fixed_formal_horizon",
                    }
                    for trace_id, duration_s in (
                        getattr(self, "_m2dgr_selection", None).excluded
                        if getattr(self, "_m2dgr_selection", None) is not None
                        else ()
                    )
                ],
                "interpolation_used": False,
                "device": str(self.device),
                "objective_bounds_identity_sha256": self.objective_bounds.identity_sha256,
                "objective_bounds_provenance_sha256": self.objective_bounds.provenance_sha256,
                "semantic_contract": dict(
                    getattr(self, "semantic_contract", {})
                ),
            },
            "metrics": aggregate_metrics,
            "runtime": {
                "total_wall_time_s": max(
                    0.0, time.monotonic() - run_started
                ),
                "training_episode_wall_time_s": sum(
                    float(record.get("episode_wall_time_s", 0.0))
                    for record in history
                ),
                "training_update_wall_time_s": sum(
                    float(record.get("training_update_wall_s", 0.0))
                    for record in history
                ),
                "policy_inference_wall_time_s": sum(
                    float(
                        record.get("metrics", {}).get(
                            "policy_inference_wall_s", 0.0
                        )
                    )
                    for record in history
                ),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if torch.device(self.device).type == "cuda"
                    else 0
                ),
            },
        }
        atomic_write_status(
            self.output_root / "job_results" / f"{self.spec.job_id}.json", result
        )

    def _complete_validation(
        self,
        completed_episode: int,
        history: list[dict],
    ) -> None:
        validation = self._validation_summary(completed_episode)
        history[-1]["validation"] = validation
        objective = float(validation["objective"])
        if objective < self.best_validation_objective:
            self.best_validation_objective = objective
            self._save_best_policy(completed_episode, validation)

    def _write_episode_progress(
        self,
        completed_episode: int,
        record: Mapping,
        *,
        run_started: float,
    ) -> None:
        elapsed_s = max(0.0, time.monotonic() - float(run_started))
        completed_training_wall_time_s = (
            float(getattr(self, "_resume_completed_wall_time_s", 0.0))
            + elapsed_s
        )
        total_episodes = int(self.spec.episodes)
        remaining_episodes = max(0, total_episodes - int(completed_episode))
        eta_s = (
            completed_training_wall_time_s
            / max(1, int(completed_episode))
            * remaining_episodes
        )
        metrics = dict(record.get("metrics", {}))
        completion_ratio = max(
            0.0,
            1.0
            - float(metrics.get("dag_drop_ratio", 0.0))
            - float(metrics.get("remaining_ratio", 0.0)),
        )
        payload = {
            "event": "episode_progress",
            "job_id": self.spec.job_id,
            "episode": int(completed_episode),
            "total_episodes": total_episodes,
            "episode_return": float(record["episode_return"]),
            "delay_weight": float(record["delay_weight"]),
            "effective_delay_mean_s": metrics.get("effective_delay_mean_s"),
            "normalized_effective_delay": metrics.get(
                "normalized_effective_delay"
            ),
            "system_energy_per_admitted_dag_j": metrics.get(
                "system_energy_per_admitted_dag_j"
            ),
            "normalized_system_energy": metrics.get("normalized_system_energy"),
            "dag_completion_ratio": completion_ratio,
            "deadline_miss_ratio": metrics.get("deadline_miss_ratio"),
            "dag_drop_ratio": metrics.get("dag_drop_ratio"),
            "task_compute_completion_ratio": metrics.get(
                "task_compute_completion_ratio"
            ),
            "actual_delivery_progress_potential": metrics.get(
                "actual_delivery_progress_potential"
            ),
            "loss": dict(record.get("loss", {})),
            "validation": record.get("validation"),
            "elapsed_s": completed_training_wall_time_s,
            "elapsed_this_process_s": elapsed_s,
            "estimated_remaining_training_s": eta_s,
        }
        atomic_write_status(
            self.output_root
            / "progress"
            / "episodes"
            / f"{self.spec.job_id}.json",
            payload,
        )
        print(json.dumps(payload, sort_keys=True), flush=True)

    def _write_micro_slot_progress(
        self,
        *,
        episode: int,
        env: UAMCOEnv,
        decision_count: int,
        episode_started: float,
        interval_started: float,
        interval_slots: int,
    ) -> None:
        now = time.monotonic()
        elapsed_s = max(0.0, now - float(episode_started))
        interval_elapsed_s = max(1.0e-9, now - float(interval_started))
        states = tuple(env.workflow_states.values())
        payload = {
            "event": "micro_slot_progress",
            "job_id": self.spec.job_id,
            "episode": int(episode),
            "micro_slot": int(env.micro_slot_index),
            "simulation_time_s": float(env.current_time_s),
            "active_dags": sum(state.status == "active" for state in states),
            "completed_dags": sum(state.status == "completed" for state in states),
            "dropped_dags": sum(state.status == "dropped" for state in states),
            "decision_count": int(decision_count),
            "episode_wall_time_s": elapsed_s,
            "recent_slot_wall_time_s": interval_elapsed_s / max(1, int(interval_slots)),
            "observed_slots_per_wall_s": int(interval_slots) / interval_elapsed_s,
        }
        atomic_write_status(
            self.output_root
            / "progress"
            / "micro_slots"
            / f"{self.spec.job_id}.json",
            payload,
        )
        print(json.dumps(payload, sort_keys=True), flush=True)

    def _load_checkpoint(self) -> tuple[int, list[dict]]:
        path = self.output_root / "checkpoints" / f"{self.spec.job_id}.pt"
        if not path.is_file():
            return 0, []
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError("checkpoint schema version does not match")
        if payload.get("job") != asdict(self.spec):
            raise RuntimeError("checkpoint job contract does not match the requested formal job")
        if payload.get("objective_bounds_identity_sha256") != self.objective_bounds.identity_sha256:
            raise RuntimeError("checkpoint frozen objective bounds identity does not match")
        if payload.get("semantic_contract") != self.semantic_contract:
            raise RuntimeError(
                "checkpoint configuration, data, or environment semantics do not match"
            )
        if self.is_domain_sota:
            if self.sota_runtime is None or "sota_runtime" not in payload:
                raise RuntimeError(f"{self.spec.method} checkpoint lacks method-specific state")
            self.sota_runtime.load_state_dict(payload["sota_runtime"])
        else:
            if self.policy is None or self.optimizer is None:
                raise RuntimeError("PPO/MARL checkpoint route is not initialized")
            self.policy.load_state_dict(payload["policy"])
            self.optimizer.load_state_dict(payload["optimizer"])
            self.lagrange.load_state_dict(payload["lagrange"])
        self.best_validation_objective = float(
            payload.get(
                "best_validation_objective",
                payload.get("best_validation_pnct", math.inf),
            )
        )
        rng_state = payload.get("rng_state")
        if not isinstance(rng_state, Mapping):
            raise RuntimeError("checkpoint lacks reproducible RNG state")
        restore_rng_state(rng_state)
        self.rng.setstate(payload["runner_rng_state"])
        return int(payload["episode"]), list(payload.get("history", ()))

    def _save_checkpoint(self, episode: int, history: Sequence[Mapping]) -> None:
        path = self.output_root / "checkpoints" / f"{self.spec.job_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
        payload = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "job": asdict(self.spec),
                "episode": int(episode),
                "best_validation_objective": self.best_validation_objective,
                "objective_bounds_identity_sha256": self.objective_bounds.identity_sha256,
                "objective_bounds_provenance_sha256": self.objective_bounds.provenance_sha256,
                "history": list(history),
                "method_route": self.method_route,
                "semantic_contract": dict(self.semantic_contract),
                "rng_state": capture_rng_state(),
                "runner_rng_state": self.rng.getstate(),
            }
        if self.is_domain_sota:
            if self.sota_runtime is None:
                raise RuntimeError("domain SOTA checkpoint route is not initialized")
            payload["sota_runtime"] = self.sota_runtime.state_dict()
        else:
            if self.policy is None or self.optimizer is None:
                raise RuntimeError("PPO/MARL checkpoint route is not initialized")
            payload.update(
                policy=self.policy.state_dict(),
                optimizer=self.optimizer.state_dict(),
                lagrange=self.lagrange.state_dict(),
            )
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _validation_summary(self, completed_training_episodes: int) -> dict:
        weights = validation_delay_weights(self.spec, self.config)
        episode_count = int(self.config["training"]["validation_episodes"])
        points: list[ParetoPoint] = []
        records: list[dict[str, float]] = []
        if self.policy is not None:
            self.policy.eval()
        try:
            with torch.no_grad():
                for weight_index, weight in enumerate(weights):
                    rollouts = [
                        self._run_episode(
                            workflows=self.fold.validation,
                            traces=self.mobility["rellis"]["validation"],
                            delay_weight=weight,
                            training=False,
                            deterministic=True,
                            episode_seed=validation_episode_seed(
                                self.spec.seed,
                                weight_index,
                                episode,
                            ),
                            experiment_mode=str(
                                self.config["evaluation"]["primary_experiment_mode"]
                            ),
                        )
                        for episode in range(episode_count)
                    ]
                    normalized_effective_delay = float(
                        mean(item.metrics["normalized_effective_delay"] for item in rollouts)
                    )
                    effective_delay_mean_s = float(
                        mean(item.metrics["effective_delay_mean_s"] for item in rollouts)
                    )
                    system_energy = float(
                        mean(
                            item.metrics["system_energy_per_admitted_dag_j"]
                            for item in rollouts
                        )
                    )
                    points.append(
                        pareto_point_from_episode_metrics(
                            {
                                "effective_delay_mean_s": effective_delay_mean_s,
                                "system_energy_per_admitted_dag_j": system_energy,
                            },
                            frozen_bounds=self.objective_bounds,
                            label=f"{weight:.2f}",
                        )
                    )
                    records.append(
                        {
                            "delay_weight": float(weight),
                            "normalized_effective_delay": normalized_effective_delay,
                            "system_energy_per_admitted_dag_j": system_energy,
                            "normalized_system_energy": float(
                                mean(item.metrics["normalized_system_energy"] for item in rollouts)
                            ),
                            "actual_delivery_progress_potential": float(
                                mean(
                                    item.metrics["actual_delivery_progress_potential"]
                                    for item in rollouts
                                )
                            ),
                            "terminal_delivery_residual_cost": float(
                                mean(
                                    item.metrics["terminal_delivery_residual_cost"]
                                    for item in rollouts
                                )
                            ),
                            "delivery_first_objective": float(
                                mean(
                                    item.metrics["delivery_first_objective"]
                                    for item in rollouts
                                )
                            ),
                        }
                    )
        finally:
            if self.policy is not None:
                self.policy.train()
        selection_kind = validation_selection_kind(self.spec)
        selection = validation_selection_objective(
            selection_kind=selection_kind,
            points=points,
            records=records,
            hypervolume_reference=(1.0, 1.0),
            residual_coefficient=1.0,
        )
        return {
            "selection_kind": selection_kind,
            **selection,
            "weights": records,
        }

    def _save_best_policy(self, episode: int, validation: Mapping) -> None:
        path = self.output_root / "checkpoints" / f"{self.spec.job_id}.best.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".pt.{os.getpid()}.tmp")
        payload = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "job": asdict(self.spec),
                "episode": int(episode),
                "validation": dict(validation),
                "method_route": self.method_route,
                "objective_bounds_identity_sha256": self.objective_bounds.identity_sha256,
                "objective_bounds_provenance_sha256": self.objective_bounds.provenance_sha256,
                "semantic_contract": dict(self.semantic_contract),
            }
        if self.is_domain_sota:
            if self.sota_runtime is None:
                raise RuntimeError("domain SOTA best-checkpoint route is not initialized")
            payload["sota_runtime"] = self.sota_runtime.state_dict()
        else:
            if self.policy is None:
                raise RuntimeError("PPO/MARL best-checkpoint route is not initialized")
            payload["policy"] = self.policy.state_dict()
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _load_best_policy(self) -> None:
        path = self.output_root / "checkpoints" / f"{self.spec.job_id}.best.pt"
        if not path.is_file():
            raise RuntimeError("formal training completed without a validation-selected checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("job") != asdict(self.spec):
            raise RuntimeError("best-checkpoint job contract does not match")
        if payload.get("semantic_contract") != self.semantic_contract:
            raise RuntimeError("best checkpoint semantics do not match")
        if self.is_domain_sota:
            if self.sota_runtime is None or "sota_runtime" not in payload:
                raise RuntimeError(f"{self.spec.method} best checkpoint lacks method state")
            self.sota_runtime.load_state_dict(payload["sota_runtime"])
        else:
            if self.policy is None:
                raise RuntimeError("PPO/MARL best-checkpoint route is not initialized")
            self.policy.load_state_dict(payload["policy"])

    def _update_policy(
        self,
        decisions: Sequence[RolloutDecision],
        admitted_count: int,
    ) -> dict[str, float]:
        if self.is_domain_sota:
            raise RuntimeError(f"{self.spec.method} must never enter the PPO update path")
        if self.policy is None or self.optimizer is None or self.ppo is None:
            raise RuntimeError("PPO/MARL update route is not initialized")
        if not decisions:
            return {"total": 0.0, "policy": 0.0, "reward_value": 0.0, "cost_value": 0.0}
        gamma = float(self.config["algorithm"]["gamma"])
        gae_lambda = float(self.config["algorithm"]["gae_lambda"])
        rewards = torch.tensor([decision.reward for decision in decisions], dtype=torch.float32)
        dones = torch.zeros(len(decisions), dtype=torch.float32)
        dones[-1] = 1.0
        reward_values = torch.tensor(
            [decision.reward_value for decision in decisions] + [0.0], dtype=torch.float32
        )
        reward_advantage, reward_return = compute_gae(
            rewards, reward_values, dones, gamma=gamma, gae_lambda=gae_lambda
        )
        if reward_advantage.numel() > 1 and float(reward_advantage.std()) > 1e-8:
            reward_advantage = (
                reward_advantage - reward_advantage.mean()
            ) / reward_advantage.std()
        cost_advantages: dict[str, Tensor] = {}
        cost_returns: dict[str, Tensor] = {}
        for key in self.cost_keys:
            signals = torch.tensor(
                [float((decision.costs or {})[key]) for decision in decisions],
                dtype=torch.float32,
            )
            values = torch.tensor(
                [decision.cost_values[key] for decision in decisions] + [0.0],
                dtype=torch.float32,
            )
            advantage, returns = compute_gae(
                signals, values, dones, gamma=gamma, gae_lambda=gae_lambda
            )
            cost_advantages[key] = advantage
            cost_returns[key] = returns

        old_log_probabilities = torch.tensor(
            [decision.old_log_probability for decision in decisions], dtype=torch.float32
        )
        chunk_size = int(self.config["algorithm"].get("graphs_per_update", 4))
        if chunk_size <= 0:
            raise ValueError("graphs per PPO update must be positive")
        epochs = ppo_epochs_for_method(self.config, self.spec.method)
        accumulated = {"total": [], "policy": [], "reward_value": [], "cost_value": []}
        indices = select_update_indices(
            len(decisions),
            int(
                self.config["algorithm"].get(
                    "max_update_decisions", 1024
                )
            ),
            self.rng,
        )
        sampled_decision_count = len(indices)
        for _ in range(epochs):
            self.rng.shuffle(indices)
            for chunk_start in range(0, len(indices), chunk_size):
                chunk = indices[chunk_start : chunk_start + chunk_size]
                index_tensor = torch.tensor(chunk, dtype=torch.long)

                def compute_chunk_losses():
                    new_log_prob, entropy, output = evaluate_rollout_decision_batch(
                        self.policy,
                        tuple(decisions[index] for index in chunk),
                        device=self.device,
                    )
                    new_reward_values = output["reward_value"]
                    new_cost_values = output["cost_values"]
                    if not isinstance(new_reward_values, Tensor) or not isinstance(
                        new_cost_values, Mapping
                    ):
                        raise TypeError("policy critics returned invalid batched values")
                    return self.ppo.compute_loss(
                        new_log_prob=new_log_prob,
                        old_log_prob=old_log_probabilities[index_tensor].to(self.device),
                        reward_advantage=reward_advantage[index_tensor].to(self.device),
                        reward_value=new_reward_values,
                        reward_return=reward_return[index_tensor].to(self.device),
                        cost_advantages={
                            key: values[index_tensor].to(self.device)
                            for key, values in cost_advantages.items()
                        },
                        cost_values=new_cost_values,
                        cost_returns={
                            key: values[index_tensor].to(self.device)
                            for key, values in cost_returns.items()
                        },
                        entropy=entropy,
                    )

                max_grad_norm = float(self.config["algorithm"]["max_grad_norm"])
                if getattr(self.policy, "update_mode", "simultaneous") == "sequential":
                    losses = self._sequential_actor_critic_step(
                        compute_chunk_losses,
                        max_grad_norm=max_grad_norm,
                    )
                else:
                    losses = compute_chunk_losses()
                    self.ppo.optimization_step(
                        self.optimizer,
                        losses.total,
                        self.policy.parameters(),
                        max_grad_norm=max_grad_norm,
                    )
                for name in accumulated:
                    accumulated[name].append(float(getattr(losses, name).detach().cpu()))
        if self.spec.variant != "without_cost_critics":
            observed = {
                key: sum(float((decision.costs or {})[key]) for decision in decisions)
                / max(1, admitted_count)
                for key in self.cost_keys
            }
            self.lagrange.update(observed)
        summary = {
            name: float(np.mean(values)) if values else 0.0
            for name, values in accumulated.items()
        }
        summary["trajectory_decisions"] = float(len(decisions))
        summary["optimized_decisions"] = float(sampled_decision_count)
        return summary

    def _sequential_actor_critic_step(self, loss_builder, *, max_grad_norm: float):
        """Apply HAPPO's fixed actor order, then update the centralized critics."""
        actor_groups = tuple(self.policy.actor_update_groups())
        all_parameters = [
            parameter for parameter in self.policy.parameters() if parameter.requires_grad
        ]
        actor_parameter_ids = {
            id(parameter) for group in actor_groups for parameter in group
        }
        critic_parameters = [
            parameter
            for parameter in all_parameters
            if id(parameter) not in actor_parameter_ids
        ]
        original_trainability = {
            id(parameter): parameter.requires_grad for parameter in self.policy.parameters()
        }

        def enable_only(enabled_parameters) -> None:
            enabled_ids = {id(parameter) for parameter in enabled_parameters}
            for parameter in self.policy.parameters():
                parameter.requires_grad_(id(parameter) in enabled_ids)

        try:
            for group in actor_groups:
                trainable_group = [
                    parameter for parameter in group if original_trainability[id(parameter)]
                ]
                enable_only(trainable_group)
                actor_losses = loss_builder()
                actor_objective = (
                    actor_losses.policy
                    - self.ppo.entropy_coefficient * actor_losses.entropy
                )
                if actor_objective.requires_grad:
                    self.ppo.optimization_step(
                        self.optimizer,
                        actor_objective,
                        trainable_group,
                        max_grad_norm=max_grad_norm,
                    )

            enable_only(critic_parameters)
            losses = loss_builder()
            critic_objective = (
                self.ppo.value_coefficient * losses.reward_value
                + self.ppo.cost_value_coefficient * losses.cost_value
            )
            self.ppo.optimization_step(
                self.optimizer,
                critic_objective,
                critic_parameters,
                max_grad_norm=max_grad_norm,
            )
            return losses
        finally:
            for parameter in self.policy.parameters():
                parameter.requires_grad_(
                    original_trainability.get(id(parameter), parameter.requires_grad)
                )

    def _evaluate(self) -> list[dict]:
        if self.policy is not None:
            self.policy.eval()
        weights = validation_delay_weights(self.spec, self.config)
        protocols = evaluation_protocols(stage=self.spec.stage, config=self.config)
        episodes_per_condition = int(self.config["evaluation"]["episodes_per_condition"])
        evaluations: list[dict] = []
        with torch.no_grad():
            for condition_index, (
                condition,
                mobility_key,
                split,
                experiment_mode,
            ) in enumerate(protocols):
                mobility = (
                    self._zero_shot_mobility()
                    if mobility_key == "m2dgr"
                    else self.mobility[mobility_key]
                )
                traces = mobility if split is None else mobility[split]
                for weight_index, delay_weight in enumerate(weights):
                    for episode in range(episodes_per_condition):
                        rollout = self._run_episode(
                            workflows=self.fold.test,
                            traces=traces,
                            delay_weight=delay_weight,
                            training=False,
                            deterministic=True,
                            episode_seed=evaluation_episode_seed(
                                self.spec.seed,
                                condition_index=condition_index,
                                weight_index=weight_index,
                                replicate_index=episode,
                            ),
                            experiment_mode=experiment_mode,
                        )
                        evaluations.append(
                            {
                                "mobility_condition": condition,
                                "experiment_mode": experiment_mode,
                                "delay_weight": delay_weight,
                                "episode": episode + 1,
                                "held_out_workflow_family": self.fold.held_out_family,
                                "metrics": rollout.metrics,
                            }
                        )
        if self.policy is not None:
            self.policy.train()
        return evaluations

    def _evaluate_smoke(self) -> list[dict]:
        delay_weight = validate_interface_scalar(
            self.config.get("objective", {}).get("interface_scalar", 0.5),
            context=(
                f"job={getattr(self.spec, 'job_id', 'unknown')} "
                f"method={getattr(self.spec, 'method', 'unknown')} "
                f"fold={getattr(self.spec, 'fold', 'unknown')} "
                f"seed={getattr(self.spec, 'seed', 'unknown')} smoke evaluation"
            ),
        )
        policy = getattr(self, "policy", None)
        if policy is not None:
            policy.eval()
        try:
            with torch.no_grad():
                rollout = self._run_episode(
                    workflows=self.fold.test,
                    traces=self.mobility["rellis"]["test"],
                    delay_weight=delay_weight,
                    training=False,
                    deterministic=True,
                    episode_seed=self.spec.seed * 1_000_000 + 91_001,
                    experiment_mode=str(
                        self.config["evaluation"][
                            "primary_experiment_mode"
                        ]
                    ),
                )
        finally:
            if policy is not None:
                policy.train()
        return [
            {
                "mobility_condition": "RELLIS-3D-test",
                "experiment_mode": str(
                    self.config["evaluation"]["primary_experiment_mode"]
                ),
                "delay_weight": delay_weight,
                "episode": 1,
                "held_out_workflow_family": getattr(
                    self.fold,
                    "held_out_family",
                    self.spec.fold
                    if hasattr(self.spec, "fold")
                    else "unknown",
                ),
                "metrics": rollout.metrics,
            }
        ]

    def _evaluate_sensitivity(self) -> list[dict]:
        main_stage = self.spec.stage == "main"
        cost_control_stage = (
            self.spec.stage == "ablation"
            and self.spec.variant == "without_cost_critics"
        )
        if not main_stage and not cost_control_stage:
            return []
        studies = self.config["evaluation"]["studies"]
        full_specifications = (
            ("scalability", "scalability_layouts", None),
            ("congestion", "workflow_trigger_distance_m", "workflow_trigger_distance_m"),
            ("contact_error", "contact_forecast_error", None),
            ("uav_failure", "unavailable_uavs", None),
            ("constraint_stress", "deadline_multiplier_scale", None),
        )
        specifications = (
            full_specifications
            if main_stage
            else (full_specifications[-1],)
        )
        episode_count = int(self.config["evaluation"]["sensitivity_episodes"])
        records: list[dict] = []
        scenario = self.config["scenario"]
        # Workflow membership is part of the preregistered nominal benchmark.
        # A scalability condition may change completion outcomes, but it must
        # not silently add or remove DAGs from the evaluation cohort.
        workflow_feasibility_config = copy.deepcopy(self.config)
        if self.policy is not None:
            self.policy.eval()
        try:
            for study_index, (study, config_key, scenario_key) in enumerate(specifications):
                for value_index, study_value in enumerate(studies.get(config_key, ())):
                    original_scenario = {
                        key: scenario[key]
                        for key in ("ugv_count", "rsu_count", "uav_count", "workflow_trigger_distance_m")
                    }
                    original_error = self._contact_forecast_error
                    original_unavailable = self._unavailable_uav_count
                    original_deadline_scale = float(
                        getattr(self, "_deadline_multiplier_scale", 1.0)
                    )
                    if study == "scalability":
                        scenario.update({key: int(value) for key, value in study_value.items()})
                        x_value = float(study_value["ugv_count"])
                    elif scenario_key:
                        scenario[scenario_key] = study_value
                        x_value = float(study_value)
                    else:
                        x_value = float(study_value)
                    if study == "contact_error":
                        self._contact_forecast_error = float(study_value)
                    if study == "uav_failure":
                        self._unavailable_uav_count = int(study_value)
                    if study == "constraint_stress":
                        self._deadline_multiplier_scale = float(study_value)
                    try:
                        for episode in range(episode_count):
                            rollout = self._run_episode(
                                workflows=self.fold.test,
                                traces=self.mobility["rellis"]["test"],
                                delay_weight=0.5,
                                training=False,
                                deterministic=True,
                                episode_seed=(
                                    self.spec.seed * 10_000_000
                                    + study_index * 1_000_000
                                    + value_index * 10_000
                                    + episode
                                ),
                                experiment_mode=(
                                    "fixed_horizon_load"
                                    if study == "congestion"
                                    else None
                                ),
                                workflow_feasibility_config=(
                                    workflow_feasibility_config
                                ),
                            )
                            records.append(
                                {
                                    "study": study,
                                    "x": x_value,
                                    "episode": episode + 1,
                                    "held_out_workflow_family": self.fold.held_out_family,
                                    "metrics": rollout.metrics,
                                }
                            )
                    finally:
                        scenario.update(original_scenario)
                        self._contact_forecast_error = original_error
                        self._unavailable_uav_count = original_unavailable
                        self._deadline_multiplier_scale = original_deadline_scale
        finally:
            if self.policy is not None:
                self.policy.train()
        return records

    @staticmethod
    def _aggregate_evaluations(evaluations: Sequence[Mapping]) -> dict[str, float]:
        metric_names = sorted(
            {
                name
                for evaluation in evaluations
                for name in evaluation.get("metrics", {})
            }
        )
        aggregated: dict[str, float] = {}
        for name in metric_names:
            values = [
                float(evaluation["metrics"][name])
                for evaluation in evaluations
                if name in evaluation.get("metrics", {})
                and math.isfinite(float(evaluation["metrics"][name]))
            ]
            aggregated[name] = float(np.mean(values)) if values else math.nan
        return aggregated
