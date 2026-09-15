from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Collection, Mapping

from .types import TaskRecord, WorkflowFold, WorkflowInstance


CONFIRMED_FAMILIES: Mapping[str, int] = {
    "Montage": 17,
    "Seismology": 11,
    "Cycles": 24,
}


def stratify_workflows_by_work(
    workflows: Collection[WorkflowInstance],
    *,
    stratum_count: int = 3,
) -> tuple[tuple[WorkflowInstance, ...], ...]:
    if stratum_count <= 0:
        raise ValueError("workload stratum count must be positive")
    ordered = sorted(
        workflows,
        key=lambda instance: (
            sum(float(task.cycles) for task in instance.tasks.values()),
            instance.instance_id,
        ),
    )
    if len(ordered) < stratum_count:
        raise ValueError("workflow corpus is smaller than the requested stratum count")
    base, remainder = divmod(len(ordered), stratum_count)
    strata: list[tuple[WorkflowInstance, ...]] = []
    start = 0
    for index in range(stratum_count):
        size = base + int(index < remainder)
        strata.append(tuple(ordered[start : start + size]))
        start += size
    return tuple(strata)


def sample_stratified_workflows(
    workflows: Collection[WorkflowInstance],
    *,
    count: int,
    rng,
) -> tuple[WorkflowInstance, ...]:
    if count <= 0:
        raise ValueError("episode workflow count must be positive")
    items = tuple(workflows)
    if not items:
        raise ValueError("cannot sample an empty workflow corpus")
    strata = stratify_workflows_by_work(
        items,
        stratum_count=min(3, len(items)),
    )
    selected: list[WorkflowInstance] = []
    for stratum in strata:
        if len(selected) >= count:
            break
        selected.append(stratum[rng.randrange(len(stratum))])
    remaining = [item for item in items if all(item is not chosen for chosen in selected)]
    while len(selected) < count:
        pool = remaining if remaining else list(items)
        index = rng.randrange(len(pool))
        selected.append(pool.pop(index))
    rng.shuffle(selected)
    return tuple(selected)


def split_stratified_validation_workflows(
    workflows: Collection[WorkflowInstance],
    *,
    validation_count: int,
) -> tuple[tuple[WorkflowInstance, ...], tuple[WorkflowInstance, ...]]:
    """Reserve validation DAGs uniformly across measured workflow scale.

    Selecting the lexicographic tail makes validation almost exclusively contain
    the largest production DAGs because WfCommons instance names encode scale.
    This fixed split instead spans the complete measured-work range and remains
    deterministic without consulting any experiment result.
    """
    ordered = sorted(
        workflows,
        key=lambda instance: (
            sum(float(task.cycles) for task in instance.tasks.values()),
            len(instance.tasks),
            instance.instance_id,
        ),
    )
    if not 0 < validation_count < len(ordered):
        raise ValueError(
            "validation count must lie strictly between zero and the family size"
        )
    if validation_count == 1:
        validation_indices = {len(ordered) // 2}
    else:
        validation_indices = {1, len(ordered) - 1}
        if validation_count > 2:
            validation_indices.update(
                round(
                    index * (len(ordered) - 1)
                    / (validation_count - 1)
                )
                for index in range(1, validation_count - 1)
            )
    if len(validation_indices) != validation_count:
        raise RuntimeError("stratified validation indices are not unique")
    validation = tuple(
        item for index, item in enumerate(ordered) if index in validation_indices
    )
    train = tuple(
        item for index, item in enumerate(ordered) if index not in validation_indices
    )
    return train, validation


def derive_task_cycles(runtime_s: float, reference_frequency_hz: float) -> float:
    runtime = float(runtime_s)
    frequency = float(reference_frequency_hz)
    if not math.isfinite(runtime) or runtime < 0:
        raise ValueError("task runtime must be finite and non-negative")
    if not math.isfinite(frequency) or frequency <= 0:
        raise ValueError("reference frequency must be finite and positive")
    return runtime * frequency


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_workflow_instance(
    path: str | Path,
    *,
    family: str | None = None,
    reference_frequency_hz: float = 1.0e9,
    source_kind: str = "production",
) -> WorkflowInstance:
    if source_kind.strip().lower() != "production":
        raise ValueError("only production execution records are accepted; WfGen inputs are forbidden")
    source_path = Path(path).resolve()
    forbidden_parts = {"wfgen", "generator", "generated"}
    if forbidden_parts.intersection(part.lower() for part in source_path.parts):
        raise ValueError("only production execution records are accepted; generator paths are forbidden")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    workflow = _require_mapping(payload.get("workflow"), "workflow")
    specification = _require_mapping(workflow.get("specification"), "workflow.specification")
    execution = _require_mapping(workflow.get("execution"), "workflow.execution")

    specification_tasks = specification.get("tasks")
    execution_tasks = execution.get("tasks")
    specification_files = specification.get("files")
    if not isinstance(specification_tasks, list) or not isinstance(execution_tasks, list):
        raise ValueError("WfFormat tasks must be lists")
    if not isinstance(specification_files, list):
        raise ValueError("WfFormat files must be a list")

    file_sizes: dict[str, int] = {}
    for item in specification_files:
        record = _require_mapping(item, "file record")
        file_id = str(record["id"])
        size = int(record["sizeInBytes"])
        if size < 0:
            raise ValueError(f"negative file size for {file_id}")
        file_sizes[file_id] = size

    executions = {
        str(_require_mapping(item, "execution task")["id"]): _require_mapping(item, "execution task")
        for item in execution_tasks
    }
    tasks: dict[str, TaskRecord] = {}
    edges: set[tuple[str, str]] = set()
    for item in specification_tasks:
        spec = _require_mapping(item, "specification task")
        task_id = str(spec["id"])
        if task_id not in executions:
            raise ValueError(f"task has no production execution record: {task_id}")
        measured = executions[task_id]
        parents = tuple(str(parent) for parent in spec.get("parents", ()))
        children = tuple(str(child) for child in spec.get("children", ()))
        input_files = tuple(str(name) for name in spec.get("inputFiles", ()))
        output_files = tuple(str(name) for name in spec.get("outputFiles", ()))
        missing_files = [name for name in input_files + output_files if name not in file_sizes]
        if missing_files:
            raise ValueError(f"task {task_id} references files without measured sizes: {missing_files}")
        runtime_s = float(measured["runtimeInSeconds"])
        tasks[task_id] = TaskRecord(
            task_id=task_id,
            name=str(spec.get("name", task_id)),
            parents=parents,
            children=children,
            input_files=input_files,
            output_files=output_files,
            runtime_s=runtime_s,
            cycles=derive_task_cycles(runtime_s, reference_frequency_hz),
            input_bytes=sum(file_sizes[name] for name in input_files),
            output_bytes=sum(file_sizes[name] for name in output_files),
            memory_bytes=int(measured.get("memoryInBytes", 0) or 0),
        )
        edges.update((parent, task_id) for parent in parents)
        edges.update((task_id, child) for child in children)

    normalized_family = (family or str(payload.get("name") or source_path.parent.name)).title()
    instance = WorkflowInstance(
        instance_id=f"{normalized_family}:{source_path.stem}",
        family=normalized_family,
        tasks=tasks,
        edges=tuple(sorted(edges)),
        source_path=source_path,
        schema_version=str(payload.get("schemaVersion", "unknown")),
        provenance={
            "runtime_system": str(_require_mapping(payload.get("runtimeSystem", {}), "runtimeSystem").get("name", "unknown")),
            "source_kind": "production",
        },
        file_sizes=file_sizes,
    )
    if not instance.tasks:
        raise ValueError(f"workflow contains no tasks: {source_path}")
    if not instance.is_acyclic():
        raise ValueError(f"workflow is not a valid DAG: {source_path}")
    return instance


def load_workflow_corpus(
    pegasus_root: str | Path,
    *,
    reference_frequency_hz: float = 1.0e9,
    enforce_confirmed_counts: bool = True,
) -> dict[str, tuple[WorkflowInstance, ...]]:
    root = Path(pegasus_root).resolve()
    corpus: dict[str, tuple[WorkflowInstance, ...]] = {}
    for family, expected_count in CONFIRMED_FAMILIES.items():
        paths = sorted((root / family.lower()).glob("*.json"))
        if enforce_confirmed_counts and len(paths) != expected_count:
            raise ValueError(
                f"{family} must contain {expected_count} production instances, found {len(paths)}"
            )
        corpus[family] = tuple(
            load_workflow_instance(
                path,
                family=family,
                reference_frequency_hz=reference_frequency_hz,
            )
            for path in paths
        )
    return corpus


def build_lofo_folds(
    corpus: Mapping[str, tuple[WorkflowInstance, ...]],
    *,
    validation_fraction: float = 0.2,
    excluded_instance_ids: Collection[str] = (),
) -> dict[str, WorkflowFold]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    normalized = {family.title(): tuple(sorted(items, key=lambda item: item.instance_id)) for family, items in corpus.items()}
    if set(normalized) != set(CONFIRMED_FAMILIES):
        raise ValueError(f"LOFO requires exactly {tuple(CONFIRMED_FAMILIES)}")
    all_ids = {
        item.instance_id
        for items in normalized.values()
        for item in items
    }
    excluded = {str(instance_id) for instance_id in excluded_instance_ids}
    missing_exclusions = sorted(excluded - all_ids)
    if missing_exclusions:
        raise ValueError(f"excluded workflow IDs do not exist: {missing_exclusions}")
    normalized = {
        family: tuple(item for item in items if item.instance_id not in excluded)
        for family, items in normalized.items()
    }
    empty_families = sorted(family for family, items in normalized.items() if not items)
    if empty_families:
        raise ValueError(f"calibration exclusion emptied workflow families: {empty_families}")

    folds: dict[str, WorkflowFold] = {}
    for held_out in CONFIRMED_FAMILIES:
        train: list[WorkflowInstance] = []
        validation: list[WorkflowInstance] = []
        for family, items in normalized.items():
            if family == held_out:
                continue
            validation_count = max(1, int(round(len(items) * validation_fraction)))
            family_train, family_validation = split_stratified_validation_workflows(
                items,
                validation_count=validation_count,
            )
            train.extend(family_train)
            validation.extend(family_validation)
        folds[held_out.lower()] = WorkflowFold(
            held_out_family=held_out,
            train=tuple(train),
            validation=tuple(validation),
            test=normalized[held_out],
        )
    for fold_name, fold in folds.items():
        leaked = sorted(
            excluded
            & {
                item.instance_id
                for split in (fold.train, fold.validation, fold.test)
                for item in split
            }
        )
        if leaked:
            raise RuntimeError(f"calibration workflow leakage in {fold_name}: {leaked}")
    return folds
