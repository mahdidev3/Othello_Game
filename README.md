# Othello Agents and Arena

This project implements a modular Othello/Reversi engine with a suite of classic search agents, a shared interface, and a tournament/benchmark harness for evaluating agents against each other.

## Project layout

```
project_root/
  src/
    agents/          # Agent implementations (reflex, minimax, alphabeta, expectimax, BFS/DFS, A*, MCTS)
    arena/           # Match/tournament/benchmark runners
    games/othello/   # Othello game rules, state, and heuristics
    network/         # Neural network helpers (policy/value network, state conversion)
    training/        # Self-play, evaluator, trainer, replay buffer
    utils/           # Timing and ANSI color helpers
  scripts/           # CLI utilities to run matches or benchmarks
  tests/             # Minimal correctness and interface tests
```

## Agents

Every agent implements the common `Agent` interface defined in `src/agents/base.py`:

* `ReflexAgent` – 1-ply heuristic chooser
* `MinimaxAgent` – depth-limited minimax
* `AlphaBetaAgent` – minimax with alpha-beta pruning
* `ExpectimaxAgent` – models the opponent as a random policy by default
* `BFSAgent` / `DFSAgent` – generic graph search with horizon limits and optional goal predicates
* `AStarAgent` – priority-queue search with pluggable heuristic
* `MonteCarloTreeSearch` – UCT-style MCTS

Othello actions are represented as `(row, col)` tuples; a pass is `None`. Deterministic tie-breaking is enforced by sorting actions.

## Running matches and tournaments

Scripts assume the repository root is on `PYTHONPATH`. Examples:

```bash
python scripts/run_match.py --agent1 alphabeta --agent2 mcts --games 20 --depth 4
python scripts/run_tournament.py --agents reflex minimax alphabeta mcts --games 4
python scripts/run_benchmark.py --agents reflex minimax alphabeta mcts expectimax --games 2 --output report.json
```

Match and tournament outputs include wins/losses/draws, average search time per move, total search time, and nodes expanded when available.

## Game and heuristics

Othello logic lives in `src/games/othello/`:

* `rules.py` – bitboard move generation, flipping, scoring, and terminal detection
* `state.py` – immutable `OthelloState` implementing the shared game protocol with a colorful `__str__`
* `heuristics.py` – mobility, parity, corner, and positional heuristics combined into a default evaluator

## Tests

Run the small regression suite with:

```bash
pytest
```

Tests cover Othello rule correctness, the agent interface, and a short arena run to ensure the harness completes without errors.
