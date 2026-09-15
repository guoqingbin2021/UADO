from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .env import SLA_TIERS, UAMCOEnv
from .queues import FiniteQueue


NODE_FEATURE_DIM = 12
BASE_GLOBAL_FEATURE_DIM = 16
ZONE_DEMAND_FEATURE_DIM = 9
GLOBAL_FEATURE_DIM = BASE_GLOBAL_FEATURE_DIM + ZONE_DEMAND_FEATURE_DIM
EXECUTOR_FEATURE_DIM = 12
EXECUTOR_CPU = 4
EXECUTOR_REACHABLE = 5
EXECUTOR_LINK_RATE = 6
EXECUTOR_QUEUE_PRESSURE = 7
EXECUTOR_INPUT_LOCALITY = 8
EXECUTOR_MISSING_INPUT = 9
EXECUTOR_CONTACT_MARGIN = 10
EXECUTOR_DELIVERY_FEASIBLE = 11


@dataclass(frozen=True, slots=True)
class GraphObservation:
    node_features: Tensor
    adjacency: Tensor
    node_mask: Tensor
    decision_mask: Tensor
    global_features: Tensor
    delay_weight: Tensor
    action_mask: Tensor
    executor_features: Tensor
    task_ids: tuple[str, ...]
    executor_actions: tuple[str, ...]

    def to(self, device: torch.device | str) -> "GraphObservation":
        return GraphObservation(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
            node_mask=self.node_mask.to(device),
            decision_mask=self.decision_mask.to(device),
            global_features=self.global_features.to(device),
            delay_weight=self.delay_weight.to(device),
            action_mask=self.action_mask.to(device),
            executor_features=self.executor_features.to(device),
            task_ids=self.task_ids,
            executor_actions=self.executor_actions,
        )


@dataclass(frozen=True, slots=True)
class GraphObservationBatch:
    node_features: Tensor
    adjacency: Tensor
    node_mask: Tensor
    decision_mask: Tensor
    global_features: Tensor
    delay_weight: Tensor
    action_mask: Tensor
    executor_features: Tensor

    def to(self, device: torch.device | str) -> "GraphObservationBatch":
        return GraphObservationBatch(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
            node_mask=self.node_mask.to(device),
            decision_mask=self.decision_mask.to(device),
            global_features=self.global_features.to(device),
            delay_weight=self.delay_weight.to(device),
            action_mask=self.action_mask.to(device),
            executor_features=self.executor_features.to(device),
        )


def batch_graph_observations(
    observations: Sequence[GraphObservation],
) -> GraphObservationBatch:
    """Pad variable-size sparse DAG views for one shared policy forward pass."""
    if not observations:
        raise ValueError("cannot batch an empty observation sequence")
    first = observations[0]
    action_count = first.action_mask.shape[-1]
    feature_dim = first.node_features.shape[-1]
    maximum_nodes = max(observation.node_features.shape[1] for observation in observations)
    batch_size = len(observations)
    node_features = torch.zeros(
        (batch_size, maximum_nodes, feature_dim), dtype=first.node_features.dtype
    )
    adjacency = torch.zeros(
        (batch_size, maximum_nodes, maximum_nodes), dtype=first.node_features.dtype
    )
    node_mask = torch.zeros((batch_size, maximum_nodes), dtype=torch.bool)
    decision_mask = torch.zeros((batch_size, maximum_nodes), dtype=torch.bool)
    action_mask = torch.zeros(
        (batch_size, maximum_nodes, action_count), dtype=torch.bool
    )
    for batch_index, observation in enumerate(observations):
        node_count = observation.node_features.shape[1]
        if observation.node_features.shape[0] != 1:
            raise ValueError("each graph observation must contain exactly one DAG")
        if (
            observation.node_features.shape[-1] != feature_dim
            or observation.action_mask.shape[-1] != action_count
            or observation.executor_actions != first.executor_actions
        ):
            raise ValueError("graph observations use incompatible feature or action spaces")
        node_features[batch_index, :node_count] = observation.node_features[0]
        node_mask[batch_index, :node_count] = observation.node_mask[0]
        decision_mask[batch_index, :node_count] = observation.decision_mask[0]
        action_mask[batch_index, :node_count] = observation.action_mask[0]
        if observation.adjacency.ndim == 2:
            if observation.adjacency.shape[0] != 2:
                raise ValueError("sparse graph adjacency must be an edge index")
            if observation.adjacency.numel():
                source, target = observation.adjacency.long()
                adjacency[batch_index, source, target] = 1.0
        elif observation.adjacency.shape == (1, node_count, node_count):
            adjacency[batch_index, :node_count, :node_count] = observation.adjacency[0]
        else:
            raise ValueError("graph observation adjacency has an invalid shape")
    return GraphObservationBatch(
        node_features=node_features,
        adjacency=adjacency,
        node_mask=node_mask,
        decision_mask=decision_mask,
        global_features=torch.cat(
            tuple(observation.global_features for observation in observations), dim=0
        ),
        delay_weight=torch.cat(
            tuple(observation.delay_weight for observation in observations), dim=0
        ),
        action_mask=action_mask,
        executor_features=torch.cat(
            tuple(observation.executor_features for observation in observations), dim=0
        ),
    )


def _queue_pressure(queues: Sequence[FiniteQueue]) -> float:
    if not queues:
        return 0.0
    ratios = []
    for queue in queues:
        item_ratio = 1.0 - queue.remaining_items / queue.max_items
        byte_ratio = 0.0 if queue.max_bytes == 0 else queue.used_bytes / queue.max_bytes
        ratios.append(max(item_ratio, byte_ratio))
    return float(sum(ratios) / len(ratios))


def select_causal_observation_task_ids(
    state,
    *,
    candidate_task_id: str,
    ready_task_ids: Sequence[str],
    max_nodes: int,
) -> tuple[str, ...]:
    """Select a deterministic candidate-centred view without truncating the environment DAG."""
    if max_nodes <= 0:
        raise ValueError("maximum observation nodes must be positive")
    instance = state.instance
    if candidate_task_id not in instance.tasks:
        raise ValueError("candidate task does not belong to the workflow")
    all_task_ids = tuple(instance.tasks)
    topological = instance.topological_indices()

    def ordered(task_ids) -> list[str]:
        return sorted(
            (task_id for task_id in task_ids if task_id in instance.tasks),
            key=lambda task_id: (topological[task_id], task_id),
        )

    candidate = instance.tasks[candidate_task_id]
    selected: set[str] = {candidate_task_id}
    for ready_task_id in ordered(ready_task_ids):
        if len(selected) >= max_nodes:
            break
        selected.add(ready_task_id)
    for neighbor_id in ordered((*candidate.parents, *candidate.children)):
        if len(selected) >= max_nodes:
            break
        selected.add(neighbor_id)
    frontier = [candidate_task_id]
    visited = {candidate_task_id}
    while frontier and len(selected) < max_nodes:
        current_id = frontier.pop(0)
        current = instance.tasks[current_id]
        neighbors = ordered((*current.parents, *current.children))
        for neighbor_id in neighbors:
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)
            frontier.append(neighbor_id)
            if len(selected) < max_nodes:
                selected.add(neighbor_id)
    return tuple(ordered(selected))


def build_graph_observation(
    env: UAMCOEnv,
    workflow_id: str,
    *,
    candidate_task_id: str | None = None,
    infrastructure_actions: Sequence[str],
    current_reachable_executors: Sequence[str],
    delay_weight: float,
    contact_history_score: float = 0.0,
    max_nodes: int | None = None,
    executor_link_rates_bps: Mapping[str, float] | None = None,
    executor_queue_pressures: Mapping[str, float] | None = None,
    executor_contact_margins: Mapping[str, float] | None = None,
    executor_delivery_feasible: Mapping[str, float] | None = None,
    enable_causal_contact_features: bool = False,
    zone_delivery_demand: Sequence[float] | None = None,
) -> GraphObservation:
    if not 0.0 <= delay_weight <= 1.0:
        raise ValueError("delay weight must lie in [0, 1]")
    state = env.workflow_states[workflow_id]
    full_task_ids = tuple(state.instance.tasks)
    if candidate_task_id is not None and candidate_task_id not in full_task_ids:
        raise ValueError("candidate task does not belong to the workflow")
    ready = set(env.precedence_ready_task_ids(workflow_id))
    task_ids = (
        full_task_ids
        if max_nodes is None or candidate_task_id is None
        else select_causal_observation_task_ids(
            state,
            candidate_task_id=candidate_task_id,
            ready_task_ids=tuple(ready),
            max_nodes=int(max_nodes),
        )
    )
    task_index = {task_id: index for index, task_id in enumerate(task_ids)}
    node_count = len(task_ids)
    full_node_count = len(full_task_ids)
    topological = state.topological_indices
    deadline_duration = max(state.deadline_time_s - state.arrival_time_s, env.micro_slot_s)
    deadline_remaining = max(0.0, state.deadline_time_s - env.current_time_s)
    tier_value = SLA_TIERS.index(state.sla_tier) / max(1, len(SLA_TIERS) - 1)
    node_rows: list[list[float]] = []
    for index, task_id in enumerate(task_ids):
        task = state.instance.tasks[task_id]
        node_rows.append(
            [
                math.log1p(task.cycles) / 30.0,
                math.log1p(task.input_bytes) / 30.0,
                math.log1p(task.output_bytes) / 30.0,
                min(2.0, task.runtime_s / max(env.episode_s, 1.0)),
                len(task.parents) / max(1, full_node_count),
                len(task.children) / max(1, full_node_count),
                float(task_id in ready),
                float(task_id in state.active_tasks),
                float(task_id in state.completed),
                min(2.0, deadline_remaining / deadline_duration),
                tier_value,
                topological[task_id] / max(1, full_node_count - 1),
            ]
        )
    edge_pairs = [
        (task_index[parent], task_index[child])
        for parent, child in state.instance.edges
        if parent in task_index and child in task_index
    ]
    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1).contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    executor_actions = ("local", *tuple(map(str, infrastructure_actions)), "defer")
    link_rates = dict(executor_link_rates_bps or {})
    queue_pressures = dict(executor_queue_pressures or {})
    contact_margins = dict(executor_contact_margins or {})
    delivery_feasible = dict(executor_delivery_feasible or {})
    masks: list[list[bool]] = []
    reachable = tuple(map(str, current_reachable_executors))
    for task_id in task_ids:
        if task_id != candidate_task_id:
            masks.append([False] * (len(executor_actions) - 1) + [True])
            continue
        raw_mask = env.action_mask(
            workflow_id,
            task_id,
            current_reachable_executors=reachable,
        )
        infrastructure_mask = [
            bool(raw_mask.get(action, False))
            for action in infrastructure_actions
        ]
        feeds_high_fan_in_sink = any(
            not state.instance.tasks[child_id].children
            and len(state.instance.tasks[child_id].parents)
            > int(env.queue_capacity_by_kind["ugv"]["max_items"])
            for child_id in state.instance.tasks[task_id].children
        )
        if enable_causal_contact_features and feeds_high_fan_in_sink:
            minimum_return_margin = (
                env.micro_slot_s / max(env.macro_interval_s, env.micro_slot_s)
            )
            infrastructure_mask = [
                allowed
                and bool(delivery_feasible.get(action, 0.0))
                and float(contact_margins.get(action, -2.0))
                >= minimum_return_margin
                for action, allowed in zip(
                    infrastructure_actions,
                    infrastructure_mask,
                )
            ]
        local_allowed = bool(raw_mask[state.owner_ugv])
        defer_allowed = bool(raw_mask["defer"])
        if (
            enable_causal_contact_features
            and not state.instance.tasks[task_id].children
            and local_allowed
        ):
            defer_allowed = False
        masks.append(
            [
                local_allowed,
                *infrastructure_mask,
                defer_allowed,
            ]
        )

    required_files = (
        state.instance.required_input_files(candidate_task_id)
        if candidate_task_id is not None
        else ()
    )
    total_required_bytes = sum(size_bytes for _, size_bytes in required_files)
    maximum_link_rate = max(float(env.link_rate_bps), 1.0)
    executor_rows: list[list[float]] = []
    for action in executor_actions:
        if action == "local":
            actual_executor = state.owner_ugv
            type_features = [1.0, 0.0, 0.0, 0.0]
            reachable_feature = 1.0
        elif action.startswith("rsu-"):
            actual_executor = action
            type_features = [0.0, 1.0, 0.0, 0.0]
            reachable_feature = float(action in reachable)
        elif action.startswith("uav-"):
            actual_executor = action
            type_features = [0.0, 0.0, 1.0, 0.0]
            reachable_feature = float(action in reachable)
        else:
            actual_executor = ""
            type_features = [0.0, 0.0, 0.0, 1.0]
            reachable_feature = 1.0
        cpu_hz = env.executor_cpu_hz.get(actual_executor, 0.0)
        missing_files = (
            state.file_ledger.missing_at(required_files, actual_executor)
            if actual_executor
            else required_files
        )
        missing_bytes = sum(size_bytes for _, size_bytes in missing_files)
        input_locality = (
            1.0
            if total_required_bytes <= 0 and actual_executor
            else (
                0.0
                if total_required_bytes <= 0
                else 1.0 - missing_bytes / total_required_bytes
            )
        )
        link_rate = (
            maximum_link_rate
            if action == "local"
            else max(0.0, float(link_rates.get(action, 0.0)))
        )
        queue_pressure = min(
            1.0,
            max(0.0, float(queue_pressures.get(action, 0.0))),
        )
        causal_margin = (
            max(-2.0, min(1.0, float(contact_margins.get(action, 0.0))))
            if enable_causal_contact_features
            else 0.0
        )
        feasible = (
            float(bool(delivery_feasible.get(action, 0.0)))
            if enable_causal_contact_features
            else 0.0
        )
        executor_rows.append(
            [
                *type_features,
                math.log1p(cpu_hz) / 30.0,
                reachable_feature,
                min(2.0, link_rate / maximum_link_rate),
                queue_pressure,
                min(1.0, max(0.0, input_locality)),
                math.log1p(missing_bytes) / 30.0,
                causal_margin,
                feasible,
            ]
        )

    reachable_ratio = len(set(reachable).intersection(infrastructure_actions)) / max(
        1, len(infrastructure_actions)
    )
    tier_one_hot = [float(state.sla_tier == tier) for tier in SLA_TIERS]
    total_cycles = sum(float(task.cycles) for task in state.instance.tasks.values())
    dependency_bytes = 0
    delivered_dependency_bytes = 0
    for parent_id, child_id in state.instance.edges:
        executor = state.scheduled.get(child_id)
        for file_id, size_bytes in state.instance.dependency_files(
            parent_id, child_id
        ):
            dependency_bytes += int(size_bytes)
            if executor is not None and state.file_ledger.has(file_id, executor):
                delivered_dependency_bytes += int(size_bytes)
    zone_demand = tuple(
        float(value)
        for value in (
            zone_delivery_demand
            if zone_delivery_demand is not None
            else (0.0,) * ZONE_DEMAND_FEATURE_DIM
        )
    )
    if len(zone_demand) != ZONE_DEMAND_FEATURE_DIM:
        raise ValueError("zone delivery demand must contain exactly nine values")
    if any(not math.isfinite(value) or value < 0.0 for value in zone_demand):
        raise ValueError("zone delivery demand must be finite and non-negative")
    global_row = [
        env.current_time_s / env.episode_s,
        deadline_remaining / max(env.episode_s, 1.0),
        float(contact_history_score),
        reachable_ratio,
        _queue_pressure(tuple(env.upload_queues.values())),
        _queue_pressure(tuple(env.compute_queues.values())),
        _queue_pressure(tuple(env.return_queues.values())),
        *tier_one_hot,
        math.log1p(total_cycles) / 30.0,
        len(ready) / max(1, full_node_count),
        len(state.scheduled) / max(1, full_node_count),
        len(state.compute_completed) / max(1, full_node_count),
        (
            1.0
            if dependency_bytes <= 0
            else delivered_dependency_bytes / dependency_bytes
        ),
        (full_node_count - node_count) / max(1, full_node_count),
        *zone_demand,
    ]
    return GraphObservation(
        node_features=torch.tensor([node_rows], dtype=torch.float32),
        adjacency=edge_index,
        node_mask=torch.ones((1, node_count), dtype=torch.bool),
        decision_mask=torch.tensor(
            [[task_id == candidate_task_id for task_id in task_ids]], dtype=torch.bool
        ),
        global_features=torch.tensor([global_row], dtype=torch.float32),
        delay_weight=torch.tensor([[delay_weight]], dtype=torch.float32),
        action_mask=torch.tensor([masks], dtype=torch.bool),
        executor_features=torch.tensor([executor_rows], dtype=torch.float32),
        task_ids=task_ids,
        executor_actions=executor_actions,
    )
