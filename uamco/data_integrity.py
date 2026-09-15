from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_record(
    project_root: str | Path,
    record: Mapping[str, object],
) -> Path:
    root = Path(project_root).resolve()
    path = (root / str(record["path"])).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"manifest path escapes project root: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)

    expected_bytes = int(record["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"file-size mismatch for {path}")

    expected_hash = str(record["sha256"])
    if not _SHA256.fullmatch(expected_hash):
        raise ValueError(f"invalid SHA-256 in manifest for {path}")
    if sha256_file(path) != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return path
