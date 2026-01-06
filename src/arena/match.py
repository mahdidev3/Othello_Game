from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from src.agents.base import Agent
from src.games.base import GameStateProtocol
from src.games.othello.state import OthelloState
from src.games.othello.rules import OthelloRules
from src.utils.colors import Colorizer


@dataclass
class MatchResult:
    winner: int
    final_state: GameStateProtocol
    moves_played: int
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)


def play_match(
    black: Agent,
    white: Agent,
    initial_state: Optional[GameStateProtocol] = None,
    verbose: bool = False,
) -> MatchResult:
    """Run a single game until termination."""

    state = initial_state or OthelloState()
    black.reset()
    white.reset()
    color = Colorizer()

    agents = {
        OthelloRules.PLAYER_BLACK: black,
        OthelloRules.PLAYER_WHITE: white,
    }
    moves = 0

    while not state.is_terminal():
        agent = agents[state.current_player]
        ssssssssssssssss = state.current_player
        agent.clear_last_search_info()
        nodes_before = agent.info().nodes_expanded
        total_time_before = agent.info().timing.total_time
        player_label = (
            color.colorize("BLACK", fg="bright_white", bg="black")
            if state.current_player == OthelloRules.PLAYER_BLACK
            else color.colorize("WHITE", fg="black", bg="bright_white")
        )

        if verbose:
            print(
                color.colorize(
                    f"\n=== Move {moves + 1} | {player_label} ({agent.name}) ===",
                    fg="cyan",
                )
            )
            print(color.colorize("Current state:", fg="yellow"))
            print(state)

        start_time = time.perf_counter()
        action = agent.select_action(state)
        elapsed = time.perf_counter() - start_time
        state = state.apply_action(action)
        moves += 1

        if verbose:
            move_text = "PASS" if action is None else f"{action}"
            print(color.colorize(f"Selected move: {move_text}", fg="green"))
            move_time = (
                elapsed
                if elapsed >= 0
                else agent.info().timing.total_time - total_time_before
            )
            print(color.colorize(f"Search time: {move_time:.6f}s", fg="yellow"))

            nodes_after = agent.info().nodes_expanded
            node_delta = nodes_after - nodes_before
            if node_delta:
                print(
                    color.colorize(
                        f"Nodes expanded this turn: {node_delta}", fg="magenta"
                    )
                )

            extras = agent.info().extra
            if extras:
                print(color.colorize("Search stats:", fg="blue"))
                for key, value in sorted(extras.items()):
                    print(color.colorize(f"  {key}: {value}", fg="blue"))

            search_info = agent.last_search_info()
            if search_info:
                print(
                    color.colorize("Algorithm insights:", fg="bright_white", bg="black")
                )
                if "value" in search_info:
                    print(
                        color.colorize(
                            f"  value: {search_info['value']:.4f}",
                            fg="bright_white",
                            bg="black",
                        )
                    )
                policy = search_info.get("policy")
                if isinstance(policy, dict) and policy:
                    top_policy = sorted(
                        policy.items(), key=lambda item: item[1], reverse=True
                    )[:5]
                    policy_str = ", ".join(
                        f"{mv}: {prob:.3f}" for mv, prob in top_policy
                    )
                    print(
                        color.colorize(
                            f"  policy (top {len(top_policy)}): {policy_str}",
                            fg="bright_white",
                            bg="black",
                        )
                    )

            print(color.colorize("Resulting state:", fg="yellow"))
            print(state)

    winner = OthelloRules.winner(state.black, state.white)  # type: ignore[attr-defined]
    stats = {
        black.name: black.info().as_dict(),
        white.name: white.info().as_dict(),
    }
    return MatchResult(
        winner=winner,
        final_state=state,
        moves_played=moves,
        stats=stats,
    )
