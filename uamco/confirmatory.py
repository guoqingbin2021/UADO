from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Mapping, Sequence

from .experiment_matrix import CHECKPOINT_SCHEMA_VERSION, ExperimentJob
from .statistics import confidence_interval_95, holm_adjust, paired_comparison


CONFIRMATORY_SEEDS = (131, 149, 167, 181, 199, 223, 251)
CONFIRMATORY_BASELINES = {
    "montage": "HAPPO",
    "seismology": "MAPPO",
    "cycles": "AMCoEdge",
}
CONFIRMATORY_EPISODES = 300
CONFIRMATORY_EVALUATION_EPISODES = 20
_SCHEDULE_ORDER = (
    ("montage", "HAPPO"),
    ("montage", "UAMCO-DAG"),
    ("cycles", "UAMCO-DAG"),
    ("seismology", "UAMCO-DAG"),
    ("seismology", "MAPPO"),
    ("cycles", "AMCoEdge"),
)
_INTEGRITY_METRICS = (
    "premature_ready_count",
    "successor_start_before_input_count",
    "byte_conservation_error_count",
    "single_server_concurrency_error_count",
)


def _slug(value: str) -> str:
    return value.lower().replace("_", "-")


def build_confirmatory_jobs(config: Mapping) -> tuple[ExperimentJob, ...]:
    section = config.get("confirmatory", {})
    registered = tuple(int(value) for value in section.get("registered_seeds", ()))
    training_seeds = tuple(int(value) for value in config["training"]["seeds"])
    if registered != CONFIRMATORY_SEEDS or training_seeds != CONFIRMATORY_SEEDS:
        raise ValueError(
            f"confirmatory registered seeds must be exactly {CONFIRMATORY_SEEDS}"
        )
    episodes = int(section.get("episodes_per_job", 0))
    if episodes != CONFIRMATORY_EPISODES:
        raise ValueError("confirmatory jobs must use exactly 300 episodes")
    evaluation_episodes = int(
        config.get("evaluation", {}).get("episodes_per_condition", 0)
    )
    if evaluation_episodes != CONFIRMATORY_EVALUATION_EPISODES:
        raise ValueError(
            "confirmatory jobs must use exactly 20 evaluation episodes per condition"
        )
    baseline_map = {
        str(fold): str(method)
        for fold, method in section.get("strongest_baseline_by_fold", {}).items()
    }
    if baseline_map != CONFIRMATORY_BASELINES:
        raise ValueError(
            "confirmatory strongest-baseline map must remain the registered mapping"
        )
    method_budgets = config["training"]["main_method_episodes"]
    for method in {"UAMCO-DAG", *CONFIRMATORY_BASELINES.values()}:
        if int(method_budgets.get(method, 0)) != CONFIRMATORY_EPISODES:
            raise ValueError(f"confirmatory method {method} must retain a 300-episode budget")

    jobs = tuple(
        ExperimentJob(
            job_id=(
                f"confirmatory__{_slug(method)}__{fold}__s{int(seed)}"
            ),
            stage="main",
            method=method,
            fold=fold,
            seed=int(seed),
            episodes=episodes,
        )
        for fold, method in _SCHEDULE_ORDER
        for seed in CONFIRMATORY_SEEDS
    )
    if len(jobs) != 42 or len({job.job_id for job in jobs}) != 42:
        raise RuntimeError("registered confirmatory matrix must contain 42 unique jobs")
    return jobs


def validate_output_isolation(
    primary_root: str | Path,
    confirmatory_root: str | Path,
) -> None:
    primary = Path(primary_root).resolve()
    confirmatory = Path(confirmatory_root).resolve()
    if (
        primary == confirmatory
        or primary in confirmatory.parents
        or confirmatory in primary.parents
    ):
        raise ValueError(
            "confirmatory output must be isolated from the frozen primary output"
        )


def _read_json(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def validate_primary_archive(
    primary_root: str | Path,
    expected_jobs: Sequence[ExperimentJob],
    *,
    expected_semantic_contract: Mapping,
) -> dict[str, object]:
    root = Path(primary_root).resolve()
    status_root = root / "status"
    result_root = root / "job_results"
    checkpoint_root = root / "checkpoints"
    if not expected_jobs:
        raise ValueError("primary archive validation requires expected jobs")

    for job in expected_jobs:
        status_path = status_root / f"{job.job_id}.json"
        result_path = result_root / f"{job.job_id}.json"
        latest_path = checkpoint_root / f"{job.job_id}.pt"
        best_path = checkpoint_root / f"{job.job_id}.best.pt"
        status = _read_json(status_path, label="primary status")
        if status.get("state") != "succeeded" or status.get("job_id") != job.job_id:
            raise RuntimeError(f"primary status is not succeeded for {job.job_id}")
        result = _read_json(result_path, label="primary result")
        if result.get("state") != "succeeded":
            raise RuntimeError(f"primary result is not succeeded for {job.job_id}")
        if result.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(f"primary result schema mismatch for {job.job_id}")
        expected_job = {**asdict(job), "protocol": "formal"}
        if result.get("job") != expected_job:
            raise RuntimeError(f"primary result job contract mismatch for {job.job_id}")
        provenance = result.get("provenance", {})
        if provenance.get("semantic_contract") != dict(expected_semantic_contract):
            raise RuntimeError(f"primary semantic contract mismatch for {job.job_id}")
        if not latest_path.is_file():
            raise RuntimeError(f"primary latest checkpoint is missing for {job.job_id}")
        if not best_path.is_file():
            raise RuntimeError(f"primary best checkpoint is missing for {job.job_id}")

    return {
        "primary_root": str(root),
        "validated_jobs": len(expected_jobs),
        "status_files": len(expected_jobs),
        "result_files": len(expected_jobs),
        "checkpoint_files": 2 * len(expected_jobs),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "semantic_contract": dict(expected_semantic_contract),
    }


def validate_confirmatory_results(
    records: Sequence[Mapping],
    jobs: Sequence[ExperimentJob],
) -> tuple[dict, ...]:
    expected = {job.job_id: job for job in jobs}
    observed: dict[str, dict] = {}
    for raw in records:
        record = dict(raw)
        job_payload = record.get("job", {})
        job_id = str(job_payload.get("job_id", ""))
        if job_id in observed:
            raise ValueError(f"duplicate confirmatory result: {job_id}")
        if job_id not in expected:
            raise ValueError(f"unexpected confirmatory result: {job_id}")
        if record.get("state") != "succeeded":
            raise ValueError(f"confirmatory result must be succeeded: {job_id}")
        if record.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"confirmatory result schema mismatch: {job_id}")
        expected_payload = {**asdict(expected[job_id]), "protocol": "formal"}
        if job_payload != expected_payload:
            raise ValueError(f"confirmatory result job contract mismatch: {job_id}")
        evaluations = [
            item
            for item in record.get("evaluation", ())
            if item.get("mobility_condition") == "RELLIS-3D-test"
            and math.isclose(float(item.get("delay_weight", -1.0)), 0.5)
        ]
        if len(evaluations) != CONFIRMATORY_EVALUATION_EPISODES:
            raise ValueError(
                "confirmatory result needs exactly "
                f"{CONFIRMATORY_EVALUATION_EPISODES} primary evaluations: {job_id}"
            )
        try:
            episode_indices = sorted(int(item["episode"]) for item in evaluations)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"confirmatory primary episode index set is invalid: {job_id}"
            ) from exc
        if episode_indices != list(
            range(1, CONFIRMATORY_EVALUATION_EPISODES + 1)
        ):
            raise ValueError(
                f"confirmatory primary episode index set is invalid: {job_id}"
            )
        for episode_index, evaluation in enumerate(evaluations):
            metrics = evaluation.get("metrics", {})
            try:
                pnct = float(metrics["pnct_mean"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"confirmatory PNCT is missing or invalid: {job_id} episode {episode_index}"
                ) from exc
            if not math.isfinite(pnct):
                raise ValueError(
                    f"confirmatory PNCT must be finite: {job_id} episode {episode_index}"
                )
            for name in _INTEGRITY_METRICS:
                try:
                    value = float(metrics[name])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"confirmatory integrity metric {name} is invalid: "
                        f"{job_id} episode {episode_index}"
                    ) from exc
                if not math.isfinite(value) or value != 0.0:
                    raise ValueError(
                        "confirmatory semantic integrity violation "
                        f"{name}={value}: {job_id} episode {episode_index}"
                    )
        observed[job_id] = record

    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"missing confirmatory results: {missing}")
    return tuple(observed[job.job_id] for job in jobs)


def _primary_evaluations(record: Mapping) -> tuple[Mapping, ...]:
    matches = [
        item
        for item in record.get("evaluation", ())
        if item.get("mobility_condition") == "RELLIS-3D-test"
        and math.isclose(float(item.get("delay_weight", -1.0)), 0.5)
    ]
    if len(matches) != CONFIRMATORY_EVALUATION_EPISODES:
        raise ValueError("validated confirmatory record lost primary evaluations")
    return tuple(matches)


def _seed_mean_pnct(record: Mapping) -> float:
    return mean(
        float(evaluation["metrics"]["pnct_mean"])
        for evaluation in _primary_evaluations(record)
    )


def build_confirmatory_report(
    records: Sequence[Mapping],
    config: Mapping,
) -> dict[str, object]:
    jobs = build_confirmatory_jobs(config)
    validated = validate_confirmatory_results(records, jobs)
    by_cell = {
        (
            str(record["job"]["fold"]),
            int(record["job"]["seed"]),
            str(record["job"]["method"]),
        ): _seed_mean_pnct(record)
        for record in validated
    }
    raw_p_values: dict[str, float] = {}
    family_rows: dict[str, dict[str, object]] = {}
    pooled_baseline: list[float] = []
    pooled_proposed: list[float] = []
    for fold, baseline in CONFIRMATORY_BASELINES.items():
        proposed = [
            by_cell[(fold, seed, "UAMCO-DAG")] for seed in CONFIRMATORY_SEEDS
        ]
        reference = [by_cell[(fold, seed, baseline)] for seed in CONFIRMATORY_SEEDS]
        comparison = paired_comparison(reference, proposed)
        differences = [left - right for left, right in zip(reference, proposed)]
        ci_low, ci_high = confidence_interval_95(differences)
        family = fold.title()
        raw_p_values[family] = comparison.p_value
        family_rows[family] = {
            "strongest_baseline": baseline,
            "n": comparison.n,
            "seeds": list(CONFIRMATORY_SEEDS),
            "proposed_pnct": proposed,
            "baseline_pnct": reference,
            "proposed_mean": mean(proposed),
            "baseline_mean": mean(reference),
            "mean_paired_improvement": mean(differences),
            "paired_improvement_ci95": [ci_low, ci_high],
            "relative_improvement": comparison.relative_improvement,
            "effect_size": comparison.effect_size,
            "test": comparison.test_name,
            "raw_p": comparison.p_value,
            "holm_adjusted_p": 1.0,
        }
        pooled_baseline.extend(reference)
        pooled_proposed.extend(proposed)
    adjusted = holm_adjust(raw_p_values)
    for family, value in adjusted.items():
        family_rows[family]["holm_adjusted_p"] = value

    pooled = paired_comparison(pooled_baseline, pooled_proposed)
    complete = len(validated) == len(jobs) == 42
    semantic_integrity = all(
        float(evaluation["metrics"][name]) == 0.0
        for record in validated
        for evaluation in _primary_evaluations(record)
        for name in _INTEGRITY_METRICS
    )
    evidence_checks = {
        "complete_registered_matrix": complete,
        "all_registered_seeds_retained": sorted(
            {int(record["job"]["seed"]) for record in validated}
        )
        == list(CONFIRMATORY_SEEDS),
        "semantic_integrity": semantic_integrity,
        "all_family_directional_advantage": all(
            float(row["relative_improvement"]) > 0.0
            for row in family_rows.values()
        ),
        "all_family_holm_significance": all(
            float(row["holm_adjusted_p"]) < 0.05
            for row in family_rows.values()
        ),
        "pooled_pnct_directional_advantage": pooled.relative_improvement > 0.0,
        "pooled_pnct_significance": pooled.p_value < 0.05,
    }
    return {
        "protocol": "pre-registered targeted confirmatory cohort",
        "registered_seeds": list(CONFIRMATORY_SEEDS),
        "episodes_per_job": CONFIRMATORY_EPISODES,
        "evaluation_episodes_per_condition": CONFIRMATORY_EVALUATION_EPISODES,
        "primary_evaluation_episode_count": (
            len(validated) * CONFIRMATORY_EVALUATION_EPISODES
        ),
        "completed_job_count": len(validated),
        "complete_registered_matrix": complete,
        "semantic_integrity": semantic_integrity,
        "families": family_rows,
        "pooled": {
            "n": pooled.n,
            "proposed_mean": mean(pooled_proposed),
            "baseline_mean": mean(pooled_baseline),
            "relative_improvement": pooled.relative_improvement,
            "effect_size": pooled.effect_size,
            "test": pooled.test_name,
            "raw_p": pooled.p_value,
        },
        "evidence_checks": evidence_checks,
        "submission_evidence_ready": all(evidence_checks.values()),
        "ccfa_acceptance_guaranteed": False,
        "interpretation_limit": (
            "This report verifies a complete, invariant-preserving confirmatory "
            "experiment. No experiment can guarantee acceptance by a CCF-A venue."
        ),
    }
