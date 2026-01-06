"""Evaluation harness for playing agents against each other."""

from .match import MatchResult, play_match
from .tournament import TournamentResult, run_tournament
from .benchmark import BenchmarkReport, run_benchmark_suite

__all__ = [
    "MatchResult",
    "TournamentResult",
    "BenchmarkReport",
    "play_match",
    "run_tournament",
    "run_benchmark_suite",
]
