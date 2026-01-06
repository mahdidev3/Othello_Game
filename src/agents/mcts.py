from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.agents.base import Agent
from src.games.base import Action, GameStateProtocol
from src.utils.timing import Timer


@dataclass(slots=True)
class SearchResult:
    value: float
    move_probabilities: Dict[Action, float]


@dataclass(slots=True)
class SearchNode:
    state: GameStateProtocol
    parent: Optional["SearchNode"]
    prior: float = 0.0

    children: Dict[Action, "SearchNode"] = field(default_factory=dict)
    untried_actions: Optional[List[Action]] = None

    visits: int = 0
    value_sum: float = 0.0

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MonteCarloTreeSearch(Agent):
    """Lightweight UCT-style Monte Carlo Tree Search that works with any GameState."""

    def __init__(
        self,
        iterations: int = 400,
        exploration_c: float = 1.4,
        rollout_limit: int = 150,
        seed: int | None = None,
    ):
        super().__init__(name="MCTS", seed=seed)
        self.iterations = int(iterations)
        self.exploration_c = float(exploration_c)
        self.rollout_limit = int(rollout_limit)
        self.search_overhead = {
            "select": {"calls": 0, "time": 0.0},
            "expand": {"calls": 0, "time": 0.0},
            "rollout": {"calls": 0, "time": 0.0},
            "backup": {"calls": 0, "time": 0.0},
        }
        self.rollout_call_totals = {
            "legal_actions": {"calls": 0, "time": 0.0},
            "apply_action": {"calls": 0, "time": 0.0},
            "is_terminal": {"calls": 0, "time": 0.0},
        }
        self.rollout_count = 0
        self.random = random.Random(seed)

    def select_action(self, state: GameStateProtocol) -> Action:
        with Timer() as timer:
            result = self.search(state)
        self._info.timing.record(timer.elapsed)
        for name, data in self.search_overhead.items():
            self._info.extra[f"{name}_time"] = data["time"]
            self._info.extra[f"{name}_calls"] = float(data["calls"])
        self._info.extra["rollouts"] = float(self.rollout_count)
        for name, data in self.rollout_call_totals.items():
            total_calls = data["calls"]
            time_taken = data["time"]
            self._info.extra[f"rollout_{name}_total_time"] = float(time_taken)
            self._info.extra[f"rollout_{name}_total_calls"] = float(total_calls)
        self.set_last_search_info(
            {"policy": result.move_probabilities, "value": result.value}
        )

        if not result.move_probabilities:
            return None
        # Deterministic tie-breaking by sorting actions.
        return max(
            result.move_probabilities.items(),
            key=lambda item: (item[1], item[0]),
        )[0]

    def _timeit(self, key: str):
        class _Timer:
            __slots__ = ("owner", "key", "t0")

            def __init__(self, owner: "MonteCarloTreeSearch", key: str):
                self.owner = owner
                self.key = key
                self.t0 = 0.0

            def __enter__(self):
                self.owner.search_overhead[self.key]["calls"] += 1
                self.t0 = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                elapsed = time.perf_counter() - self.t0
                self.owner.search_overhead[self.key]["time"] += elapsed

        return _Timer(self, key)

    def search(self, root_state: GameStateProtocol) -> SearchResult:
        for data in self.search_overhead.values():
            data["calls"] = 0
            data["time"] = 0.0
        self.rollout_count = 0
        for key in self.rollout_call_totals:
            self.rollout_call_totals[key]["calls"] = 0
            self.rollout_call_totals[key]["time"] = 0.0

        root = SearchNode(state=root_state, parent=None)
        root.untried_actions = list(root_state.legal_actions())

        if not root.untried_actions:
            return SearchResult(value=root_state.outcome(), move_probabilities={})

        for _ in range(self.iterations):
            node = self._select(root)
            with self._timeit("expand"):
                node = self._expand(node)
            with self._timeit("rollout"):
                value = self._rollout(node.state)
            with self._timeit("backup"):
                self._backup(node, value)

        move_probs = self._distribution_from_root(root)
        return SearchResult(
            value=float(max(-1.0, min(1.0, root.q_value()))),
            move_probabilities=move_probs,
        )

    def _select(self, node: SearchNode) -> SearchNode:
        with self._timeit("select"):
            while True:
                if node.untried_actions is None:
                    node.untried_actions = list(node.state.legal_actions())
                if node.untried_actions:
                    return node
                if not node.children:
                    return node

                best_child: Optional[SearchNode] = None
                best_score = -1e18
                for mv, child in node.children.items():
                    score = self._ucb_score(node.visits, child)
                    if score > best_score:
                        best_score = score
                        best_child = child
                if best_child is None:
                    return node
                node = best_child

    def _expand(self, node: SearchNode) -> SearchNode:
        if node.untried_actions is None:
            node.untried_actions = list(node.state.legal_actions())
        if node.untried_actions:
            mv = node.untried_actions.pop()
            child_state = node.state.apply_action(mv)
            child = SearchNode(state=child_state, parent=node)
            node.children[mv] = child
            return child
        return node

    def _rollout(self, state: GameStateProtocol) -> float:
        cur = state
        start_player = state.current_player
        steps = 0
        passes = 0
        local_counts = {
            "legal_actions": {"calls": 0, "time": 0.0},
            "apply_action": {"calls": 0, "time": 0.0},
            "is_terminal": {"calls": 0, "time": 0.0},
        }

        while True:
            t0 = time.perf_counter()
            is_term = cur.is_terminal()
            elapsed = time.perf_counter() - t0
            local_counts["is_terminal"]["calls"] += 1
            local_counts["is_terminal"]["time"] += elapsed
            if is_term or steps >= self.rollout_limit:
                break
            t0 = time.perf_counter()
            moves = cur.legal_actions()
            elapsed = time.perf_counter() - t0
            local_counts["legal_actions"]["calls"] += 1
            local_counts["legal_actions"]["time"] += elapsed
            if not moves:
                t0 = time.perf_counter()
                cur = cur.apply_action(None)
                elapsed = time.perf_counter() - t0
                local_counts["apply_action"]["calls"] += 1
                local_counts["apply_action"]["time"] += elapsed
                passes += 1
                if passes >= 2:
                    break
                continue

            passes = 0
            mv = self._biased_choice(moves)
            t0 = time.perf_counter()
            cur = cur.apply_action(mv)
            elapsed = time.perf_counter() - t0
            local_counts["apply_action"]["calls"] += 1
            local_counts["apply_action"]["time"] += elapsed
            steps += 1
            self._info.nodes_expanded += 1

        self.rollout_count += 1
        for key, counts in local_counts.items():
            self.rollout_call_totals[key]["calls"] += counts["calls"]
            self.rollout_call_totals[key]["time"] += counts["time"]

        return cur.outcome(perspective=start_player)

    def _biased_choice(self, moves: List[Action]) -> Action:
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
        for c in sorted(corners):
            if c in moves:
                return c
        return self.random.choice(moves)

    def _backup(self, node: SearchNode, value: float) -> None:
        v = value
        cur: Optional[SearchNode] = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += v
            v = -v
            cur = cur.parent

    def _ucb_score(self, parent_visits: int, child: SearchNode) -> float:
        exploration = self.exploration_c * math.sqrt(
            math.log(parent_visits + 1.0) / (child.visits + 1e-9)
        )
        return child.q_value() + exploration

    @staticmethod
    def _distribution_from_root(root: SearchNode) -> Dict[Action, float]:
        total_visits = sum(child.visits for child in root.children.values())
        if total_visits <= 0:
            return {}
        return {mv: child.visits / total_visits for mv, child in root.children.items()}
