from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .data_validation import validate_formal_data
from .experiment_matrix import ABLATION_VARIANTS, FORMAL_METHODS


@dataclass(frozen=True, slots=True)
class TrainingJobSpec:
    job_id: str
    stage: str
    method: str
    fold: str
    seed: int
    episodes: int
    delay_weight: float | None = None
    variant: str | None = None
    protocol: str = "formal"

    def __post_init__(self) -> None:
        if not self.job_id or self.stage not in {"main", "ablation"}:
            raise ValueError("invalid formal training job identifier or stage")
        if self.method not in FORMAL_METHODS:
            raise ValueError(f"unknown formal method: {self.method}")
        if self.fold not in {"montage", "seismology", "cycles"}:
            raise ValueError(f"unknown LOFO workflow fold: {self.fold}")
        if self.protocol not in {"formal", "smoke"}:
            raise ValueError(f"unknown training protocol: {self.protocol}")
        if self.protocol == "formal" and not 20 <= self.episodes <= 10_000:
            raise ValueError(
                "formal jobs require twenty to ten thousand training episodes"
            )
        if self.protocol == "smoke" and not 1 <= self.episodes <= 20:
            raise ValueError("smoke jobs require one to twenty training episodes")
        if self.delay_weight is not None and self.delay_weight != 0.5:
            raise ValueError(
                "the delivery-first protocol accepts only the legacy interface scalar 0.5"
            )
        if self.stage == "ablation" and self.variant not in ABLATION_VARIANTS:
            raise ValueError("ablation jobs require one registered retraining variant")
        if self.stage == "main" and self.variant is not None:
            raise ValueError("main jobs cannot set an ablation variant")

    @property
    def uses_continuous_preferences(self) -> bool:
        return False


class ContinuousPreferenceSampler:
    def __init__(self, minimum: float, maximum: float, *, seed: int) -> None:
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("continuous preference interval must lie inside [0, 1]")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self._random = random.Random(int(seed))
        self._pending: list[float] = []

    def sample(self) -> float:
        if not self._pending:
            span = self.maximum - self.minimum
            strata = list(range(5))
            self._random.shuffle(strata)
            self._pending = [
                self.minimum
                + span * (stratum + self._random.random()) / len(strata)
                for stratum in strata
            ]
        return self._pending.pop(0)

    def get_state(self):
        return {
            "random": self._random.getstate(),
            "pending": tuple(self._pending),
        }

    def set_state(self, state) -> None:
        if not isinstance(state, Mapping):
            raise ValueError("preference sampler state must be a mapping")
        self._random.setstate(state["random"])
        self._pending = [float(value) for value in state["pending"]]


def run_training_job(config: Mapping, spec: TrainingJobSpec) -> int:
    """Run one validated job without substituting data or fabricated metrics."""
    project_root = Path(__file__).resolve().parents[1]
    workflow_manifest = project_root / str(config["data"]["workflow_manifest"])
    mobility_manifest = project_root / str(config["data"]["mobility_manifest"])
    report = validate_formal_data(
        project_root=project_root,
        workflow_manifest=workflow_manifest,
        mobility_manifest=mobility_manifest,
        calibration_manifest=project_root / str(config["calibration"]["manifest"]),
        objective_bounds=project_root / str(config["calibration"]["frozen_bounds"]),
        active_config=config if spec.protocol == "formal" else None,
    )
    report.require_ready()
    from .formal_engine import FormalExperimentRunner

    runner = FormalExperimentRunner(config=config, spec=spec, project_root=project_root)
    runner.run()
    return 0
