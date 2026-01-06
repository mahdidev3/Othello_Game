from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.games.base import Action, GameStateProtocol
from src.utils.timing import TimingStats


@dataclass
class AgentInfo:
    """Container for metrics reported by an agent after play."""

    timing: TimingStats = field(default_factory=TimingStats)
    nodes_expanded: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        data = {"nodes_expanded": float(self.nodes_expanded), **self.timing.as_dict()}
        data.update(self.extra)
        return data


class Agent(ABC):
    """Shared interface for all agents."""

    name: str

    def __init__(self, name: Optional[str] = None, seed: Optional[int] = None) -> None:
        self.name = name or self.__class__.__name__
        if seed is not None:
            random.seed(seed)
        self._info = AgentInfo()

    @abstractmethod
    def select_action(self, state: GameStateProtocol) -> Action:
        """Return the action chosen for the provided state."""
        raise NotImplementedError

    def reset(self) -> None:
        """Hook for resetting agent state between games."""
        self._info = AgentInfo()

    def info(self) -> AgentInfo:
        """Return metrics collected so far."""
        return self._info
