from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "UAMCO-DAG": "#0072B2",
    "MAPPO": "#E69F00",
    "HAPPO": "#56B4E9",
    "AMCoEdge": "#D55E00",
    "FDEdge": "#009E73",
    "MEC-UARA": "#CC79A7",
}
METHOD_MARKERS = {
    "UAMCO-DAG": "o",
    "MAPPO": "v",
    "HAPPO": "P",
    "AMCoEdge": "s",
    "FDEdge": "^",
    "MEC-UARA": "D",
}
DISPLAY_METHOD_NAMES = {"UAMCO-DAG": "UADO"}


def _display_method(method: str) -> str:
    return DISPLAY_METHOD_NAMES.get(str(method), str(method))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(figure: plt.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError("publication figures must use vector PDF output")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    return path


def plot_metric_comparison(
    method_statistics: Mapping[str, Mapping[str, float]],
    *,
    y_label: str,
    output_path: str | Path,
) -> Path:
    _style()
    methods = tuple(method_statistics)
    means = [float(method_statistics[method]["mean"]) for method in methods]
    intervals = [float(method_statistics[method].get("ci95", 0.0)) for method in methods]
    figure, axis = plt.subplots(figsize=(3.5, 2.35))
    positions = np.arange(len(methods))
    axis.bar(
        positions,
        means,
        yerr=intervals,
        color=[METHOD_COLORS.get(method, "#777777") for method in methods],
        edgecolor="black",
        linewidth=0.5,
        capsize=2.5,
    )
    axis.set_xticks(positions, [_display_method(method) for method in methods], rotation=18, ha="right")
    axis.set_ylabel(y_label)
    axis.set_axisbelow(True)
    return _save(figure, output_path)


def plot_pareto_fronts(
    method_points: Mapping[str, Sequence[tuple[float, float]]],
    *,
    output_path: str | Path,
) -> Path:
    _style()
    figure, axis = plt.subplots(figsize=(3.5, 2.45))
    for method, points in method_points.items():
        ordered = sorted((float(delay), float(energy)) for delay, energy in points)
        axis.plot(
            [point[0] for point in ordered],
            [point[1] for point in ordered],
            color=METHOD_COLORS.get(method, "#777777"),
            marker=METHOD_MARKERS.get(method, "o"),
            markersize=3.5,
            linewidth=1.2,
            label=_display_method(method),
        )
    axis.set_xlabel("Normalized penalized completion time")
    axis.set_ylabel("Normalized mobile energy")
    axis.legend(frameon=False)
    axis.set_axisbelow(True)
    return _save(figure, output_path)


def plot_line_study(
    series: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    x_label: str,
    y_label: str,
    output_path: str | Path,
) -> Path:
    _style()
    figure, axis = plt.subplots(figsize=(3.5, 2.35))
    for method, values in series.items():
        x_values = np.asarray(values["x"], dtype=float)
        means = np.asarray(values["mean"], dtype=float)
        intervals = np.asarray(values.get("ci95", np.zeros_like(means)), dtype=float)
        color = METHOD_COLORS.get(method, "#777777")
        axis.plot(
            x_values,
            means,
            color=color,
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=1.2,
            markersize=3.5,
            label=_display_method(method),
        )
        axis.fill_between(x_values, means - intervals, means + intervals, color=color, alpha=0.15)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(frameon=False)
    axis.set_axisbelow(True)
    return _save(figure, output_path)


def plot_energy_breakdown(
    breakdown: Mapping[str, Mapping[str, float]],
    *,
    output_path: str | Path,
) -> Path:
    _style()
    methods = tuple(breakdown)
    components = ("UGV compute", "UGV radio", "UAV compute", "UAV radio", "UAV propulsion")
    positions = np.arange(len(methods))
    bottom = np.zeros(len(methods))
    figure, axis = plt.subplots(figsize=(3.5, 2.5))
    palette = ("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
    for component, color in zip(components, palette):
        values = np.asarray([breakdown[method].get(component, 0.0) for method in methods], dtype=float)
        axis.bar(positions, values, bottom=bottom, label=component, color=color, linewidth=0)
        bottom += values
    axis.set_xticks(positions, [_display_method(method) for method in methods], rotation=18, ha="right")
    axis.set_ylabel("Energy per admitted DAG (J)")
    axis.legend(frameon=False, ncol=2)
    axis.set_axisbelow(True)
    return _save(figure, output_path)


def plot_all_publication_figures(payload: Mapping, output_dir: str | Path) -> tuple[Path, ...]:
    root = Path(output_dir)
    generated: list[Path] = []
    for metric, label in (
        ("pnct", "Penalized normalized completion time"),
        ("deadline_miss", "Deadline-miss ratio"),
        ("dag_drop", "DAG-drop ratio"),
        ("throughput", "Workflow throughput (DAG/s)"),
    ):
        if metric in payload.get("comparisons", {}):
            generated.append(
                plot_metric_comparison(
                    payload["comparisons"][metric],
                    y_label=label,
                    output_path=root / f"{metric}_comparison.pdf",
                )
            )
    if payload.get("pareto"):
        generated.append(plot_pareto_fronts(payload["pareto"], output_path=root / "pareto_front.pdf"))
    for key, x_label in (
        ("scalability", "Number of UGVs"),
        ("congestion", "Workflow trigger distance (m)"),
        ("contact_error", "Contact forecast error"),
        ("uav_failure", "Unavailable UAVs"),
        ("ablation", "Ablation variant"),
    ):
        if key in payload:
            generated.append(
                plot_line_study(
                    payload[key],
                    x_label=x_label,
                    y_label="Penalized normalized completion time",
                    output_path=root / f"{key}.pdf",
                )
            )
    if payload.get("energy_breakdown"):
        generated.append(
            plot_energy_breakdown(payload["energy_breakdown"], output_path=root / "energy_breakdown.pdf")
        )
    return tuple(generated)
