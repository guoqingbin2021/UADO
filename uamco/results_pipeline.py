from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .experiment_matrix import atomic_write_status
from .plots import plot_all_publication_figures
from .publication_gate import validate_publication_results
from .result_aggregation import aggregate_formal_records


PLACEHOLDER_MACROS = (
    "FormalResultStatus",
    "BestPNCTImprovement",
    "BestHVImprovement",
    "BestMissReduction",
    "BestDropReduction",
)


def write_latex_results_macros(
    payload: Mapping[str, float] | None,
    output_path: str | Path,
    *,
    gate_passed: bool,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r"\newif\ifformalresultspassed"]
    if not gate_passed or payload is None:
        lines.append(r"\formalresultspassedfalse")
        lines.extend(rf"\newcommand{{\{name}}}{{--}}" for name in PLACEHOLDER_MACROS)
    else:
        lines.append(r"\formalresultspassedtrue")
        lines.append(r"\newcommand{\FormalResultStatus}{passed}")
        lines.append(
            rf"\newcommand{{\BestPNCTImprovement}}{{{float(payload['pnct_improvement_percent']):.2f}\%}}"
        )
        lines.append(
            rf"\newcommand{{\BestHVImprovement}}{{{float(payload['hv_improvement_percent']):.2f}\%}}"
        )
        lines.append(
            rf"\newcommand{{\BestMissReduction}}{{{float(payload['miss_reduction_percent']):.2f}\%}}"
        )
        lines.append(
            rf"\newcommand{{\BestDropReduction}}{{{float(payload['drop_reduction_percent']):.2f}\%}}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_job_results(results_dir: str | Path) -> tuple[dict, ...]:
    root = Path(results_dir)
    records = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("state") == "succeeded" and "metrics" in payload:
            records.append(payload)
    return tuple(records)


def postprocess_formal_results(config: Mapping) -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / config["runtime"]["output_dir"]
    macros_path = output_root / "results_macros.tex"
    results = load_job_results(output_root / "job_results")
    if not results:
        write_latex_results_macros(None, macros_path, gate_passed=False)
        atomic_write_status(
            output_root / "gate_report.json",
            {
                "state": "not_run",
                "passed": False,
                "allow_advantage_claims": False,
                "reason": "No completed formal job results were found.",
            },
        )
        return 2

    atomic_write_status(output_root / "aggregated_results.json", {"jobs": results})
    try:
        aggregated = aggregate_formal_records(results, config)
    except (KeyError, ValueError) as exc:
        write_latex_results_macros(None, macros_path, gate_passed=False)
        atomic_write_status(
            output_root / "gate_report.json",
            {
                "state": "incomplete",
                "passed": False,
                "allow_advantage_claims": False,
                "reason": str(exc),
            },
        )
        return 3
    atomic_write_status(output_root / "publication_gate_input.json", aggregated.gate_input)
    atomic_write_status(output_root / "statistical_report.json", aggregated.statistical_report)
    atomic_write_status(output_root / "paper_summary.json", aggregated.summary)
    atomic_write_status(output_root / "plot_payload.json", aggregated.plot_payload)
    report = validate_publication_results(aggregated.gate_input)
    atomic_write_status(
        output_root / "gate_report.json",
        {
            "state": "complete",
            "passed": report.passed,
            "allow_advantage_claims": report.allow_advantage_claims,
            "checks": dict(report.checks),
            "failures": report.failures,
        },
    )
    write_latex_results_macros(aggregated.summary, macros_path, gate_passed=report.passed)
    plot_all_publication_figures(aggregated.plot_payload, output_root / "figures")
    return 0 if report.passed else 4
