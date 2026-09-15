from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol


class QueueItem(Protocol):
    item_id: str

    @property
    def storage_bytes(self) -> int: ...


@dataclass(slots=True)
class TransferItem:
    item_id: str
    workflow_id: str
    task_id: str
    source: str
    destination: str
    total_bytes: int
    remaining_bytes: float
    sla_tier: str = "silver"
    deadline_s: float = math.inf
    direction: str = "upload"
    paused_s: float = 0.0
    resume_count: int = 0
    is_paused: bool = False
    file_id: str | None = None
    producer_task_id: str | None = None
    consumer_task_id: str | None = None
    final_destination: str | None = None
    route: tuple[str, ...] = ()
    hop_index: int = 0
    attempt: int = 1
    wasted_bytes: float = 0.0
    has_started: bool = False
    has_been_interrupted: bool = False
    source_energy_j: float = 0.0
    destination_energy_j: float = 0.0

    def __post_init__(self) -> None:
        if self.total_bytes < 0 or not 0 <= self.remaining_bytes <= self.total_bytes:
            raise ValueError("transfer byte counts are invalid")

    @property
    def storage_bytes(self) -> int:
        return int(math.ceil(self.remaining_bytes))

    @property
    def complete(self) -> bool:
        return self.remaining_bytes <= 0


@dataclass(slots=True)
class ComputeItem:
    item_id: str
    workflow_id: str
    task_id: str
    executor: str
    total_cycles: float
    remaining_cycles: float
    sla_tier: str = "silver"
    deadline_s: float = math.inf
    result_bytes: int = 0
    remaining_critical_path_s: float = 0.0
    topological_index: int = 0
    enqueue_time_s: float = 0.0
    data_ready_time_s: float = math.inf
    compute_start_time_s: float | None = None
    compute_complete_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.total_cycles < 0 or not 0 <= self.remaining_cycles <= self.total_cycles:
            raise ValueError("compute cycle counts are invalid")
        if self.result_bytes < 0:
            raise ValueError("result size cannot be negative")

    @property
    def storage_bytes(self) -> int:
        return int(self.result_bytes)

    @property
    def complete(self) -> bool:
        return self.remaining_cycles <= 0


class FiniteQueue:
    def __init__(self, *, max_items: int, max_bytes: int) -> None:
        if max_items <= 0 or max_bytes < 0:
            raise ValueError("queue capacities are invalid")
        self.max_items = int(max_items)
        self.max_bytes = int(max_bytes)
        self._items: OrderedDict[str, QueueItem] = OrderedDict()

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    @property
    def used_bytes(self) -> int:
        return sum(item.storage_bytes for item in self._items.values())

    @property
    def remaining_items(self) -> int:
        return self.max_items - len(self._items)

    @property
    def remaining_bytes(self) -> int:
        return self.max_bytes - self.used_bytes

    def enqueue(self, item: QueueItem) -> bool:
        if item.item_id in self._items:
            raise ValueError(f"duplicate queue item: {item.item_id}")
        if len(self._items) >= self.max_items:
            return False
        if self.used_bytes + item.storage_bytes > self.max_bytes:
            return False
        self._items[item.item_id] = item
        return True

    def get(self, item_id: str) -> QueueItem:
        return self._items[item_id]

    def remove(self, item_id: str) -> QueueItem:
        return self._items.pop(item_id)

    def items(self) -> tuple[QueueItem, ...]:
        return tuple(self._items.values())


def total_buffered_bytes(queues: tuple[FiniteQueue, ...] | list[FiniteQueue]) -> int:
    return sum(queue.used_bytes for queue in queues)
