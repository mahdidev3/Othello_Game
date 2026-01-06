from __future__ import annotations

import math
from typing import Dict, List, Optional

from .interfaces import BaseMCTS, GameState, SearchNode, SearchResult


class MonteCarloTreeSearch(BaseMCTS):
    """
    Lightweight UCT-style Monte Carlo Tree Search that works with any GameState.

    The implementation keeps allocations minimal and mirrors the timing hooks used
    elsewhere in the project for transparent logging.
    """

    def __init__(
        self, iterations: int = 800, exploration_c: float = 1.4, rollout_limit: int = 200
    ):
        super().__init__(iterations=iterations, exploration_c=exploration_c)
        self.rollout_limit = int(rollout_limit)

    def search(self, root_state: GameState) -> SearchResult:
        root = SearchNode(state=root_state, parent=None)
        root.untried_moves = list(root_state.legal_moves())

        if not root.untried_moves:
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
                if node.untried_moves is None:
                    node.untried_moves = list(node.state.legal_moves())
                if node.untried_moves:
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
        if node.untried_moves is None:
            node.untried_moves = list(node.state.legal_moves())
        if node.untried_moves:
            mv = node.untried_moves.pop()
            child_state = node.state.apply_move(mv)
            child = SearchNode(state=child_state, parent=node)
            node.children[mv] = child
            return child
        return node

    def _rollout(self, state: GameState) -> float:
        cur = state
        start_player = state.current_player
        steps = 0
        passes = 0

        while not cur.is_terminal() and steps < self.rollout_limit:
            moves = cur.legal_moves()
            if not moves:
                cur = cur.apply_move(None)
                passes += 1
                if passes >= 2:
                    break
                continue

            passes = 0
            mv = self._biased_choice(moves)
            cur = cur.apply_move(mv)
            steps += 1

        return cur.outcome(perspective=start_player)

    @staticmethod
    def _biased_choice(moves: List[int]) -> int:
        corners = (0, 7, 56, 63)
        for c in corners:
            if c in moves:
                return c

        h = 0x9E3779B97F4A7C15
        for m in moves:
            h ^= (m + 0x9E3779B9) + ((h << 6) & ((1 << 64) - 1)) + (h >> 2)
        idx = (h ^ (h >> 33)) % len(moves)
        return moves[int(idx)]

    def _backup(self, node: SearchNode, value: float) -> None:
        v = value
        cur: Optional[SearchNode] = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += v
            v = -v
            cur = cur.parent

    @staticmethod
    def _distribution_from_root(root: SearchNode) -> Dict[int, float]:
        total_visits = sum(child.visits for child in root.children.values())
        if total_visits <= 0:
            return {}
        return {mv: child.visits / total_visits for mv, child in root.children.items()}

