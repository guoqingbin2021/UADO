from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .calibration import load_calibration_contract
from .data_integrity import verify_file_record
from .mobility_data import load_m2dgr_ground_truth, load_rellis_pose_trace
from .workflow_data import (
    CONFIRMED_FAMILIES,
    build_lofo_folds,
    load_workflow_instance,
)


@dataclass(frozen=True, slots=True)
class FormalDataReport:
    ready: bool
    workflow_instance_count: int
    workflow_parsed_count: int
    mobility_trace_count: int
    verified_file_count: int
    calibration_leakage_count: int
    missing_files: tuple[str, ...]
    errors: tuple[str, ...]

    def require_ready(self) -> None:
        if not self.ready:
            details = "; ".join(self.errors) if self.errors else "required files are missing"
            raise RuntimeError(
                f"formal real-data preflight failed: {details}; missing={list(self.missing_files)}"
            )


def _read_manifest(path: str | Path) -> Mapping:
    resolved = Path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"manifest root must be a mapping: {resolved}")
    return payload


def validate_formal_data(
    *,
    project_root: str | Path,
    workflow_manifest: str | Path,
    mobility_manifest: str | Path,
    calibration_manifest: str | Path | None = None,
    objective_bounds: str | Path | None = None,
    active_config: Mapping[str, Any] | None = None,
) -> FormalDataReport:
    """Check the exact production-workflow and measured-pose contract without loading training tensors."""
    root = Path(project_root).resolve()
    workflow_payload = _read_manifest(workflow_manifest)
    mobility_payload = _read_manifest(mobility_manifest)
    errors: list[str] = []
    missing: list[str] = []
    workflow_count = 0
    workflow_parsed_count = 0
    mobility_count = 0
    verified_count = 0
    calibration_leakage_count = 0
    parsed_corpus: dict[str, list] = {family: [] for family in CONFIRMED_FAMILIES}

    if workflow_payload.get("source_kind") != "production_execution_records":
        errors.append("workflow source_kind must be production_execution_records")
    if workflow_payload.get("generator_allowed", True):
        errors.append("workflow generators, including WfGen, must be disabled")
    families = workflow_payload.get("families")
    workflow_files = workflow_payload.get("files")
    if not isinstance(families, Mapping) or set(families) != {"Montage", "Seismology", "Cycles"}:
        errors.append("workflow manifest must contain exactly Montage, Seismology, and Cycles")
    else:
        for family, record in families.items():
            if not isinstance(record, Mapping):
                errors.append(f"invalid workflow family record: {family}")
                continue
            folder = root / str(record.get("path", ""))
            expected = int(record.get("instances", -1))
            paths = tuple(sorted(folder.glob("*.json"))) if folder.is_dir() else ()
            if len(paths) != expected:
                errors.append(f"{family} requires {expected} JSON records, found {len(paths)}")
                if not folder.is_dir():
                    missing.append(folder.as_posix())
    if not isinstance(workflow_files, Mapping):
        errors.append("workflow manifest must contain a per-file files mapping")
    else:
        workflow_count = len(workflow_files)
        if workflow_count != sum(CONFIRMED_FAMILIES.values()):
            errors.append(f"workflow files mapping requires 52 records, found {workflow_count}")
        records = list(workflow_files.items())
        listed_paths = [
            str(record.get("path", "")) if isinstance(record, Mapping) else ""
            for _, record in records
        ]
        if listed_paths != sorted(listed_paths):
            errors.append("workflow file records must be ordered by stable project-relative path")
        if len(set(listed_paths)) != len(listed_paths):
            errors.append("workflow file records contain duplicate paths")
        family_record_counts = {family: 0 for family in CONFIRMED_FAMILIES}
        verified_paths: set[Path] = set()
        for instance_id, record in records:
            if not isinstance(record, Mapping):
                errors.append(f"invalid workflow file record: {instance_id}")
                continue
            family = str(record.get("family", ""))
            if family not in CONFIRMED_FAMILIES:
                errors.append(f"workflow {instance_id} has invalid family {family}")
                continue
            family_record_counts[family] += 1
            raw_path = str(record.get("path", ""))
            if "\\" in raw_path or Path(raw_path).is_absolute():
                errors.append(f"workflow path must be project-relative POSIX: {raw_path}")
            expected_folder = (root / str(families[family]["path"])).resolve()
            try:
                path = verify_file_record(root, record)
                if path.parent != expected_folder:
                    raise ValueError(f"workflow path/family mismatch for {instance_id}: {path}")
                if path in verified_paths:
                    raise ValueError(f"duplicate workflow path: {path}")
                instance = load_workflow_instance(
                    path,
                    family=family,
                    reference_frequency_hz=1.0e9,
                )
                if instance.instance_id != str(instance_id):
                    raise ValueError(
                        f"workflow manifest ID mismatch: {instance_id} != {instance.instance_id}"
                    )
                if instance.schema_version != "1.5":
                    raise ValueError(
                        f"unsupported WfFormat schema {instance.schema_version} for {instance_id}"
                    )
                parsed_corpus[family].append(instance)
                verified_paths.add(path)
                verified_count += 1
                workflow_parsed_count += 1
            except FileNotFoundError as exc:
                missing.append(Path(exc.filename or raw_path).resolve().as_posix())
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"workflow {instance_id}: {exc}")
        for family, expected in CONFIRMED_FAMILIES.items():
            if family_record_counts[family] != expected:
                errors.append(
                    f"workflow files mapping requires {expected} {family} records, "
                    f"found {family_record_counts[family]}"
                )
            folder = (root / str(families[family]["path"])).resolve()
            actual_paths = set(folder.glob("*.json")) if folder.is_dir() else set()
            recorded_paths = {
                path for path in verified_paths if path.parent == folder
            }
            if actual_paths != recorded_paths:
                errors.append(f"workflow {family} directory and files mapping differ")

    if not mobility_payload.get("pose_only", False):
        errors.append("mobility manifest must be pose_only")
    if not mobility_payload.get("no_interpolation", False):
        errors.append("mobility manifest must forbid interpolation")
    if mobility_payload.get("schema_version") != 2:
        errors.append("mobility manifest must use schema version 2")
    rellis = mobility_payload.get("rellis3d", {})
    splits = rellis.get("splits", {}) if isinstance(rellis, Mapping) else {}
    sampling_hz = rellis.get("sampling_hz") if isinstance(rellis, Mapping) else None
    if sampling_hz != 10.0:
        errors.append("RELLIS-3D sampling_hz must be 10.0")
    if not isinstance(rellis, Mapping) or rellis.get("time_axis") != "sensor_rate":
        errors.append("RELLIS-3D time_axis must be sensor_rate")
    if set(splits) != {"calibration", "train", "validation", "test"}:
        errors.append("RELLIS-3D requires disjoint calibration/train/validation/test splits")
    else:
        seen: set[str] = set()
        rellis_root = root / str(rellis.get("local_root", ""))
        rellis_files = rellis.get("files", {})
        expected_rellis = {str(value) for values in splits.values() for value in values}
        if not isinstance(rellis_files, Mapping) or set(rellis_files) != expected_rellis:
            errors.append("RELLIS-3D manifest file records are incomplete or contain extras")
        for split, sequence_ids in splits.items():
            if not sequence_ids:
                errors.append(f"RELLIS-3D {split} split is empty")
            for sequence_id in sequence_ids:
                sequence = str(sequence_id)
                if sequence in seen:
                    errors.append(f"RELLIS-3D split leakage for sequence {sequence}")
                seen.add(sequence)
                mobility_count += 1
                expected_path = (rellis_root / sequence / "poses.txt").resolve()
                record = rellis_files.get(sequence) if isinstance(rellis_files, Mapping) else None
                if not isinstance(record, Mapping):
                    missing.append(expected_path.as_posix())
                    continue
                try:
                    path = verify_file_record(root, record)
                    if path != expected_path:
                        raise ValueError(
                            f"RELLIS-3D record path mismatch for {sequence}: {path}"
                        )
                    load_rellis_pose_trace(
                        path,
                        trace_id=sequence,
                        split=str(split),
                        sampling_hz=float(sampling_hz),
                    )
                    verified_count += 1
                except FileNotFoundError as exc:
                    missing.append(Path(exc.filename or expected_path).resolve().as_posix())
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"RELLIS-3D {sequence}: {exc}")
        if tuple(str(value) for value in splits.get("calibration", ())) != ("00000",):
            errors.append("RELLIS-3D calibration split must contain exactly sequence 00000")

    m2dgr = mobility_payload.get("m2dgr_outdoor", {})
    if not isinstance(m2dgr, Mapping) or m2dgr.get("training_allowed", True):
        errors.append("M2DGR Outdoor must be zero-shot only")
    else:
        m2dgr_root = root / str(m2dgr.get("local_root", ""))
        m2dgr_files = m2dgr.get("files", {})
        expected_m2dgr = {str(value).lower() for value in m2dgr.get("sequences", ())}
        if not isinstance(m2dgr_files, Mapping) or set(m2dgr_files) != expected_m2dgr:
            errors.append("M2DGR manifest file records are incomplete or contain extras")
        for sequence_id in m2dgr.get("sequences", ()):
            sequence = str(sequence_id).lower()
            mobility_count += 1
            expected_path = (m2dgr_root / sequence / "ground_truth.txt").resolve()
            record = m2dgr_files.get(sequence) if isinstance(m2dgr_files, Mapping) else None
            if not isinstance(record, Mapping):
                missing.append(expected_path.as_posix())
                continue
            try:
                path = verify_file_record(root, record)
                if path != expected_path:
                    raise ValueError(f"M2DGR record path mismatch for {sequence}: {path}")
                load_m2dgr_ground_truth(path, trace_id=sequence)
                verified_count += 1
            except FileNotFoundError as exc:
                missing.append(Path(exc.filename or expected_path).resolve().as_posix())
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"M2DGR {sequence}: {exc}")

    if calibration_manifest is None:
        errors.append("formal data validation requires a calibration manifest")
    else:
        try:
            contract = load_calibration_contract(
                calibration_manifest,
                project_root=root,
                objective_bounds_path=objective_bounds,
                active_config=active_config,
            )
            selected = set(contract.workflow_instance_ids)
            available = {
                instance.instance_id
                for items in parsed_corpus.values()
                for instance in items
            }
            if not selected <= available:
                raise ValueError(
                    f"calibration workflows are absent from corpus: {sorted(selected - available)}"
                )
            expected_smallest = {
                min(items, key=lambda item: (len(item.tasks), item.source_path.as_posix())).instance_id
                for items in parsed_corpus.values()
                if items
            }
            if selected != expected_smallest:
                raise ValueError(
                    "calibration workflows must be the stable smallest instance from each family"
                )
            if set(str(value) for value in splits.get("calibration", ())) != {
                contract.mobility_trace_id
            }:
                raise ValueError("calibration workflow and mobility selections disagree")
            folds = build_lofo_folds(
                {family: tuple(items) for family, items in parsed_corpus.items()},
                excluded_instance_ids=contract.workflow_instance_ids,
            )
            for fold in folds.values():
                used = {
                    item.instance_id
                    for split_items in (fold.train, fold.validation, fold.test)
                    for item in split_items
                }
                calibration_leakage_count += len(selected & used)
            formal_trace_ids = {
                str(value)
                for split_name, values in splits.items()
                if split_name != "calibration"
                for value in values
            }
            calibration_leakage_count += int(contract.mobility_trace_id in formal_trace_ids)
            if calibration_leakage_count:
                errors.append(
                    f"calibration leakage detected in formal splits: {calibration_leakage_count}"
                )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"calibration contract: {exc}")

    unique_missing = tuple(sorted(set(missing)))
    return FormalDataReport(
        ready=(
            not errors
            and not unique_missing
            and workflow_count == workflow_parsed_count == 52
            and verified_count == workflow_parsed_count + mobility_count
            and calibration_leakage_count == 0
        ),
        workflow_instance_count=workflow_count,
        workflow_parsed_count=workflow_parsed_count,
        mobility_trace_count=mobility_count,
        verified_file_count=verified_count,
        calibration_leakage_count=calibration_leakage_count,
        missing_files=unique_missing,
        errors=tuple(errors),
    )
