from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    name: str
    parents: tuple[str, ...]
    children: tuple[str, ...]
    input_files: tuple[str, ...]
    output_files: tuple[str, ...]
    runtime_s: float
    cycles: float
    input_bytes: int
    output_bytes: int
    memory_bytes: int = 0


@dataclass(slots=True)
class WorkflowInstance:
    instance_id: str
    family: str
    tasks: Mapping[str, TaskRecord]
    edges: tuple[tuple[str, str], ...]
    source_path: Path
    schema_version: str
    provenance: Mapping[str, str] = field(default_factory=dict)
    file_sizes: Mapping[str, int] = field(default_factory=dict)

    def is_acyclic(self) -> bool:
        indegree = {task_id: 0 for task_id in self.tasks}
        children: dict[str, list[str]] = {task_id: [] for task_id in self.tasks}
        for parent, child in self.edges:
            if parent not in indegree or child not in indegree:
                return False
            indegree[child] += 1
            children[parent].append(child)
        ready = [task_id for task_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            task_id = ready.pop()
            visited += 1
            for child in children[task_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        return visited == len(self.tasks)

    def ready_task_ids(
        self,
        completed: set[str] | frozenset[str],
        scheduled: set[str] | frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        return tuple(
            task_id
            for task_id, task in self.tasks.items()
            if task_id not in completed
            and task_id not in scheduled
            and all(parent in completed for parent in task.parents)
        )

    def topological_indices(self) -> Mapping[str, int]:
        """Return deterministic Kahn ranks, breaking simultaneous ties by ID."""
        indegree = {task_id: 0 for task_id in self.tasks}
        children = {task_id: [] for task_id in self.tasks}
        for parent_id, child_id in self.edges:
            indegree[child_id] += 1
            children[parent_id].append(child_id)
        ready = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        indices: dict[str, int] = {}
        while ready:
            task_id = ready.pop(0)
            indices[task_id] = len(indices)
            for child_id in sorted(children[task_id]):
                indegree[child_id] -= 1
                if indegree[child_id] == 0:
                    ready.append(child_id)
            ready.sort()
        if len(indices) != len(self.tasks):
            raise ValueError("workflow graph must be acyclic")
        return indices

    def dependency_files(
        self,
        parent_id: str,
        child_id: str,
    ) -> tuple[tuple[str, int], ...]:
        parent_outputs = set(self.tasks[parent_id].output_files)
        child_inputs = set(self.tasks[child_id].input_files)
        shared_files = parent_outputs & child_inputs
        return tuple(
            sorted((file_id, int(self.file_sizes[file_id])) for file_id in shared_files)
        )

    def external_input_files(self, task_id: str) -> tuple[tuple[str, int], ...]:
        produced_files = {
            file_id
            for task in self.tasks.values()
            for file_id in task.output_files
        }
        external_files = set(self.tasks[task_id].input_files) - produced_files
        return tuple(
            sorted((file_id, int(self.file_sizes[file_id])) for file_id in external_files)
        )

    def required_input_files(self, task_id: str) -> tuple[tuple[str, int], ...]:
        return tuple(
            sorted(
                (file_id, int(self.file_sizes[file_id]))
                for file_id in self.tasks[task_id].input_files
            )
        )

    def sink_task_ids(self) -> tuple[str, ...]:
        return tuple(
            task_id
            for task_id, task in self.tasks.items()
            if not task.children
        )

    def remaining_critical_path_s(self, task_id: str) -> float:
        memo: dict[str, float] = {}

        def visit(current_id: str) -> float:
            if current_id in memo:
                return memo[current_id]
            task = self.tasks[current_id]
            downstream = max((visit(child) for child in task.children), default=0.0)
            memo[current_id] = float(task.runtime_s) + downstream
            return memo[current_id]

        return visit(task_id)


@dataclass(frozen=True, slots=True)
class WorkflowFold:
    held_out_family: str
    train: tuple[WorkflowInstance, ...]
    validation: tuple[WorkflowInstance, ...]
    test: tuple[WorkflowInstance, ...]


@dataclass(frozen=True, slots=True)
class MobilityTrace:
    trace_id: str
    source: str
    split: str
    timestamps_s: tuple[float, ...]
    positions_xy_m: tuple[tuple[float, float], ...]
    provenance: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioLayout:
    traces: tuple[MobilityTrace, ...]
    rsu_positions_xy_m: tuple[tuple[float, float], ...]
    uav_initial_positions_xyz_m: tuple[tuple[float, float, float], ...]
    width_m: float
    height_m: float
