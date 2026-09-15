from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m uamco.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--protocol", choices=("formal", "smoke"), default="formal")
    train.add_argument("--job-id", required=True)
    train.add_argument("--stage", required=True)
    train.add_argument("--method", required=True)
    train.add_argument("--fold", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--episodes", type=int, required=True)
    train.add_argument("--delay-weight", type=float)
    train.add_argument("--variant")

    postprocess = subparsers.add_parser("postprocess")
    postprocess.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate-config":
        from .data_validation import validate_formal_data

        project_root = Path(__file__).resolve().parents[1]
        report = validate_formal_data(
            project_root=project_root,
            workflow_manifest=project_root / config["data"]["workflow_manifest"],
            mobility_manifest=project_root / config["data"]["mobility_manifest"],
            calibration_manifest=project_root / config["calibration"]["manifest"],
            objective_bounds=project_root / config["calibration"]["frozen_bounds"],
            active_config=config,
        )
        report.require_ready()
        print(
            json.dumps(
                {
                    "valid": True,
                    "config": str(Path(args.config).resolve()),
                    "formal_data": asdict(report),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "train":
        from .training import TrainingJobSpec, run_training_job

        spec = TrainingJobSpec(
            job_id=args.job_id,
            stage=args.stage,
            method=args.method,
            fold=args.fold,
            seed=args.seed,
            episodes=args.episodes,
            delay_weight=args.delay_weight,
            variant=args.variant,
            protocol=args.protocol,
        )
        return run_training_job(config, spec)
    if args.command == "postprocess":
        from .results_pipeline import postprocess_formal_results

        return postprocess_formal_results(config)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
