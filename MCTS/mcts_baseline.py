from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

from .othello_game import OthelloGame


@dataclass(slots=True)
class _Node:
    state: OthelloGame
    parent: Optional["_Node"]
    prior: float = 0.0

    children: Dict[int, "_Node"] = field(default_factory=dict)  # move -> node
    untried: Optional[List[int]] = None

    N: int = 0
    W: float = 0.0  # total value from THIS node player's perspective

    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


class MCTSBaseline:
    """
    Fast pure MCTS (UCT) for Othello using the bitboard OthelloGame.

    - No neural net guidance (that's handled by MCTSModel elsewhere).
    - Designed for ~1000 iterations per move in Python quickly by:
        * bitboard game state (very fast legal moves + apply)
        * minimal allocations in hot paths
        * shallow Node structure (slots)
    """

    def __init__(self, iterations: int = 800, exploration_c: float = 1.4, rollout_limit: int = 200):
        self.iterations = int(iterations)
        self.c = float(exploration_c)
        self.rollout_limit = int(rollout_limit)

        # instrumentation compatible with your SelfPlayGame report
        self.search_overhead = {
            "select": {"calls": 0, "time": 0.0},
            "expand": {"calls": 0, "time": 0.0},
            "rollout": {"calls": 0, "time": 0.0},
            "backup": {"calls": 0, "time": 0.0},
        }

    def _timeit(self, key: str):
        class _T:
            __slots__ = ("outer", "key", "t0")
            def __init__(self, outer, key):
                self.outer = outer
                self.key = key
                self.t0 = 0.0
            def __enter__(self):
                self.outer.search_overhead[self.key]["calls"] += 1
                self.t0 = time.perf_counter()
            def __exit__(self, exc_type, exc, tb):
                self.outer.search_overhead[self.key]["time"] += (time.perf_counter() - self.t0)
        return _T(self, key)

    def search(self, root_state: OthelloGame) -> Tuple[float, Dict[int, float]]:
        """
        Returns:
          value_estimate: float in [-1,1] from perspective of root_state._player
          move_probs: {move: prob} from visit counts at root
        """
        root = _Node(state=root_state, parent=None)
        root.untried = root_state.legal_moves()

        # Edge case: no moves
        if not root.untried:
            return 0.0, {}

        for _ in range(self.iterations):
            node = root

            # -------- Selection --------
            with self._timeit("select"):
                while True:
                    if node.untried is None:
                        node.untried = node.state.legal_moves()

                    if node.untried:
                        break  # expandable
                    if not node.children:
                        break  # terminal or dead-end

                    # UCT choose best child
                    log_N = math.log(node.N + 1.0)
                    best_score = -1e18
                    best_child = None
                    for mv, ch in node.children.items():
                        # UCB1: Q + c*sqrt(log(N)/n)
                        u = self.c * math.sqrt(log_N / (ch.N + 1e-9))
                        score = ch.Q() + u
                        if score > best_score:
                            best_score = score
                            best_child = ch
                    if best_child is None:
                        break
                    node = best_child

            # -------- Expansion --------
            with self._timeit("expand"):
                if node.untried is None:
                    node.untried = node.state.legal_moves()

                if node.untried:
                    mv = node.untried.pop()  # pop last (fast)
                    child_state = node.state.make_move(mv)
                    child = _Node(state=child_state, parent=node)
                    child.untried = None  # lazy init
                    node.children[mv] = child
                    node = child

            # -------- Rollout / Simulation --------
            with self._timeit("rollout"):
                value = self._rollout(node.state)  # from perspective of node.state._player

            # -------- Backpropagation --------
            with self._timeit("backup"):
                self._backup(node, value)

        # Build move distribution from root visit counts
        total_visits = sum(ch.N for ch in root.children.values())
        if total_visits <= 0:
            return 0.0, {}

        move_probs: Dict[int, float] = {mv: ch.N / total_visits for mv, ch in root.children.items()}

        # Root value estimate: average value from root player perspective
        value_est = root.Q()
        return float(max(-1.0, min(1.0, value_est))), move_probs

    def _rollout(self, state: OthelloGame) -> float:
        """
        Fast random rollout from `state`.

        Returns outcome from perspective of `state._player`:
          +1 win, -1 loss, 0 draw
        """
        cur = state
        start_player = cur._player

        steps = 0
        passes = 0
        while not cur.is_terminal() and steps < self.rollout_limit:
            moves = cur.legal_moves()
            if not moves:
                cur = cur.make_move(None)
                passes += 1
                if passes >= 2:
                    break
                continue
            passes = 0
            # simple fast policy: choose midgame moves slightly preferring corners
            mv = self._biased_choice(moves)
            cur = cur.make_move(mv)
            steps += 1

        winner = cur.check_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == start_player else -1.0

    @staticmethod
    def _biased_choice(moves: List[int]) -> int:
        """
        Small speed-friendly bias that helps strength without costing much:
        - prefer corners if available
        - otherwise uniform random-ish via xorshift on hash
        """
        corners = (0, 7, 56, 63)
        for c in corners:
            for m in moves:
                if m == c:
                    return c

        # tiny deterministic pseudo-random based on current list contents
        # (avoids importing random in hot loop)
        h = 0x9E3779B97F4A7C15
        for m in moves:
            h ^= (m + 0x9e3779b9) + ((h << 6) & ((1<<64)-1)) + (h >> 2)
        idx = (h ^ (h >> 33)) % len(moves)
        return moves[int(idx)]

    def _backup(self, node: _Node, value: float) -> None:
        """
        Back up `value` up the tree.
        `value` is from perspective of `node.state._player`. When moving to parent, flip sign.
        """
        v = value
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += v
            v = -v
            cur = cur.parent
