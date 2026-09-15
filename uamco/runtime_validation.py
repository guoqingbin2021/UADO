"""Canonical event-trace validation with incremental replay support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RuntimeInvariantReport:
    successor_start_before_input_count: int
    transfer_byte_error_count: int
    overlapping_compute_count: int

    @property
    def passed(self) -> bool:
        return (
            self.successor_start_before_input_count == 0
            and self.transfer_byte_error_count == 0
            and self.overlapping_compute_count == 0
        )


class RuntimeTraceValidator:
    """Replay only newly appended events, retaining placement and service state."""

    def __init__(self) -> None:
        self._placements: set[tuple[str, str, str]] = set()
        self._active_by_executor: dict[str, tuple[str, str, str]] = {}
        self._transfer_progress: dict[tuple[str, str, int, int, str], float] = {}
        self._transfer_expected: dict[tuple[str, str, int, int, str], int] = {}
        self._successor_errors = 0
        self._byte_errors = 0
        self._overlap_errors = 0
        self.processed_event_count = 0

    @property
    def report(self) -> RuntimeInvariantReport:
        return RuntimeInvariantReport(
            successor_start_before_input_count=self._successor_errors,
            transfer_byte_error_count=self._byte_errors,
            overlapping_compute_count=self._overlap_errors,
        )

    @staticmethod
    def _time(event: Mapping[str, object]) -> float:
        return float(event["time_s"])

    @staticmethod
    def _required_inputs(event: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        canonical = event.get("required_inputs")
        if canonical is not None:
            return tuple(canonical)
        normalized = []
        for required in event.get("required_input_files", ()):
            if isinstance(required, str):
                normalized.append({"file_id": required})
            else:
                file_id, size = required
                normalized.append({"file_id": file_id, "expected_bytes": size})
        return tuple(normalized)

    @staticmethod
    def _transfer_key(event: Mapping[str, object]) -> tuple[str, str, int, int, str]:
        """Identify one physical transfer hop without trusting delivered bytes."""
        return (
            str(event["workflow_id"]),
            str(event["file_id"]),
            int(event.get("attempt", 1)),
            int(event.get("hop_index", 0)),
            str(event.get("transfer_id", "")),
        )

    def consume(self, events: Sequence[Mapping[str, object]]) -> RuntimeInvariantReport:
        """Process one micro-slot's new events in timestamp order."""
        for event in sorted(events, key=self._time):
            self.processed_event_count += 1
            kind = event.get("event")
            if kind == "file_placement":
                self._placements.add(
                    (str(event["workflow_id"]), str(event["file_id"]), str(event["executor"]))
                )
            elif kind == "transfer_progress":
                transfer_key = self._transfer_key(event)
                expected = int(event["expected_bytes"])
                previous_expected = self._transfer_expected.get(transfer_key)
                if previous_expected is not None and previous_expected != expected:
                    self._byte_errors += 1
                self._transfer_progress[transfer_key] = self._transfer_progress.get(
                    transfer_key, 0.0
                ) + float(event["transferred_bytes"])
                self._transfer_expected[transfer_key] = expected
            elif kind == "file_delivery":
                transfer_key = self._transfer_key(event)
                expected = int(event["expected_bytes"])
                delivered = float(event.get("delivered_bytes", event.get("bytes", 0.0)))
                progressed = self._transfer_progress.pop(transfer_key, 0.0)
                progress_expected = self._transfer_expected.pop(transfer_key, None)
                if (
                    delivered != expected
                    or (
                        expected != 0
                        and (
                            progress_expected != expected
                            or not math.isclose(
                                progressed,
                                delivered,
                                rel_tol=1.0e-12,
                                abs_tol=1.0e-6,
                            )
                        )
                    )
                    or (expected == 0 and abs(progressed) > 1.0e-9)
                ):
                    self._byte_errors += 1
            elif kind == "compute_start":
                workflow_id = str(event["workflow_id"])
                executor = str(event["executor"])
                for required in self._required_inputs(event):
                    file_id = str(required["file_id"])
                    if (workflow_id, file_id, executor) not in self._placements:
                        self._successor_errors += 1
                        break
                service_id = (workflow_id, str(event["task_id"]), executor)
                if executor in self._active_by_executor:
                    self._overlap_errors += 1
                else:
                    self._active_by_executor[executor] = service_id
            elif kind in {"compute_complete", "compute_cancel"}:
                executor = str(event["executor"])
                service_id = (str(event["workflow_id"]), str(event["task_id"]), executor)
                if self._active_by_executor.get(executor) == service_id:
                    self._active_by_executor.pop(executor)
        return self.report


def validate_event_trace(
    events: Sequence[Mapping[str, object]],
) -> RuntimeInvariantReport:
    """One-shot chronological replay for smoke/debug checks."""
    validator = RuntimeTraceValidator()
    return validator.consume(events)
