from __future__ import annotations

from collections.abc import Iterable, Mapping


class FileLedger:
    """Tracks which executor currently stores each workflow file."""

    def __init__(self, file_sizes: Mapping[str, int]) -> None:
        self._file_sizes = {str(file_id): int(size) for file_id, size in file_sizes.items()}
        self._locations = {file_id: set() for file_id in self._file_sizes}

    def place(self, file_id: str, node_id: str) -> None:
        self._locations[file_id].add(str(node_id))

    def has(self, file_id: str, node_id: str) -> bool:
        return str(node_id) in self._locations[file_id]

    def locations(self, file_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._locations[file_id]))

    def missing_at(
        self,
        required_files: Iterable[tuple[str, int]],
        node_id: str,
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            (file_id, int(size))
            for file_id, size in required_files
            if not self.has(file_id, node_id)
        )

    def local_reuse_bytes(
        self,
        required_files: Iterable[tuple[str, int]],
        node_id: str,
    ) -> int:
        return sum(
            int(size)
            for file_id, size in required_files
            if self.has(file_id, node_id)
        )
