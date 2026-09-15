from __future__ import annotations

import hashlib
import math
from typing import Mapping


Link = tuple[str, str]


def _validate_outage_arguments(
    *, slot: int, outage_probability: float, burst_slots: int
) -> None:
    if int(slot) != slot or slot < 0:
        raise ValueError("slot must be a nonnegative integer")
    if not math.isfinite(float(outage_probability)) or not (
        0.0 <= float(outage_probability) < 1.0
    ):
        raise ValueError("outage_probability must lie in [0, 1)")
    if int(burst_slots) != burst_slots or burst_slots <= 0:
        raise ValueError("burst_slots must be a positive integer")


def _canonical_link(link: Link) -> Link:
    if len(link) != 2 or not all(isinstance(endpoint, str) and endpoint for endpoint in link):
        raise ValueError("link must contain two nonempty endpoint identifiers")
    first, second = sorted(link)
    return first, second


def burst_link_available(
    *,
    episode_seed: int,
    link: Link,
    slot: int,
    outage_probability: float,
    burst_slots: int,
) -> bool:
    """Return whether a geometry-valid link survives a paired burst outage."""

    _validate_outage_arguments(
        slot=slot,
        outage_probability=outage_probability,
        burst_slots=burst_slots,
    )
    if float(outage_probability) == 0.0:
        return True
    first, second = _canonical_link(link)
    block = int(slot) // int(burst_slots)
    payload = f"{int(episode_seed)}|{first}|{second}|{block}".encode("utf-8")
    sample = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(1 << 64)
    return sample >= float(outage_probability)


def apply_burst_outage(
    connectivity: Mapping[Link, bool],
    *,
    episode_seed: int,
    slot: int,
    outage_probability: float,
    burst_slots: int,
) -> tuple[dict[Link, bool], int, int]:
    """Mask geometry-valid links and return masked links, eligible count, and removals."""

    _validate_outage_arguments(
        slot=slot,
        outage_probability=outage_probability,
        burst_slots=burst_slots,
    )
    masked: dict[Link, bool] = {}
    eligible = 0
    removed = 0
    for link, connected in connectivity.items():
        if not connected:
            masked[link] = False
            continue
        eligible += 1
        available = burst_link_available(
            episode_seed=episode_seed,
            link=link,
            slot=slot,
            outage_probability=outage_probability,
            burst_slots=burst_slots,
        )
        masked[link] = available
        removed += int(not available)
    return masked, eligible, removed

