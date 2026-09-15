from __future__ import annotations

import json
import hashlib
from pathlib import Path

from .amcoedge import AMCoEdgeRuntime
from .fdedge import FDEdgeRuntime
from .mec_uara import MECUARARuntime


SOURCE_CONTRACT = {
    "AMCoEdge": {
        "commit": "fd921f8e11f7b3ca1cf0b4a7baeffb45d0a7ffe6",
        "mode": "faithful_pytorch_port",
        "files": ("AdaDQN.py", "environment.py", "main.py"),
        "runtime": AMCoEdgeRuntime,
    },
    "FDEdge": {
        "commit": "551988866f934b5cdb672e9122279a73eba60259",
        "mode": "package_safe_core_port",
        "files": ("fdedge_main.py", "fdsac_model.py", "feedback_diffusion.py"),
        "runtime": FDEdgeRuntime,
    },
    "MEC-UARA": {
        "commit": "304a60a0a9db5d0a41242e150dca6e383b1c68c7",
        "mode": "direct_primal_dual_port",
        "files": ("agent.py", "env.py", "main.py"),
        "runtime": MECUARARuntime,
    },
}


def _validate_source_record(method: str, project_root: str | Path | None) -> None:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    manifest_path = root / "third_party/SOURCES.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{method} cannot start: third_party/SOURCES.json is missing")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload.get("methods", {}).get(method)
    if not isinstance(record, dict):
        raise RuntimeError(f"{method} cannot start: exact source record is missing")
    expected = SOURCE_CONTRACT[method]
    if record.get("commit") != expected["commit"]:
        raise RuntimeError(f"{method} cannot start: pinned source commit differs from adapter")
    adaptation = record.get("adaptation", {})
    if adaptation.get("mode") != expected["mode"]:
        raise RuntimeError(f"{method} cannot start: declared adapter mode differs from executable core")
    source_root = (root / str(record.get("local_path", ""))).resolve()
    third_party_root = (root / "third_party").resolve()
    if not source_root.is_relative_to(third_party_root):
        raise RuntimeError(f"{method} cannot start: source path escapes third_party")
    missing = [name for name in expected["files"] if not (source_root / name).is_file()]
    if missing:
        raise RuntimeError(f"{method} cannot start: pinned source files are missing: {missing}")
    expected_hashes = record.get("source_sha256")
    if not isinstance(expected_hashes, dict):
        raise RuntimeError(f"{method} cannot start: source SHA-256 records are missing")
    mismatched = []
    for name in expected["files"]:
        expected_hash = expected_hashes.get(name)
        actual_hash = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        if expected_hash != actual_hash:
            mismatched.append(name)
    if mismatched:
        raise RuntimeError(
            f"{method} cannot start: vendored source hash mismatch: {mismatched}"
        )


def build_sota_runtime(name: str, **kwargs):
    try:
        contract = SOURCE_CONTRACT[name]
    except KeyError as exc:
        supported = ", ".join(SOURCE_CONTRACT)
        raise ValueError(f"unsupported domain SOTA runtime {name!r}; expected one of {supported}") from exc
    project_root = kwargs.pop("project_root", None)
    _validate_source_record(name, project_root)
    return contract["runtime"](**kwargs)
