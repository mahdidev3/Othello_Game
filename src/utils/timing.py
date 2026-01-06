from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator


class Timer:
    """Simple context manager for measuring elapsed time."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.elapsed = time.perf_counter() - self.start


@dataclass
class TimingStats:
    """Track aggregated timing metrics for agents."""

    total_time: float = 0.0
    move_count: int = 0
    per_phase: Dict[str, float] = field(default_factory=dict)

    def record(self, duration: float, phase: str | None = None) -> None:
        self.total_time += duration
        self.move_count += 1
        if phase:
            self.per_phase[phase] = self.per_phase.get(phase, 0.0) + duration

    def average(self) -> float:
        return self.total_time / self.move_count if self.move_count else 0.0

    def as_dict(self) -> Dict[str, float]:
        data = {
            "total_time": self.total_time,
            "moves": float(self.move_count),
            "avg_time": self.average(),
        }
        for phase, elapsed in self.per_phase.items():
            data[f"phase_{phase}"] = elapsed
        return data


def time_block() -> Iterator[Timer]:
    timer = Timer()
    with timer:
        yield timer
