from src.agents import ReflexAgent
from src.arena.match import play_match
from src.arena.tournament import run_tournament


def test_match_completes():
    black = ReflexAgent()
    white = ReflexAgent()
    result = play_match(black, white)
    assert result.moves_played > 0
    assert black.name in result.stats
    assert white.name in result.stats


def test_tournament_summary():
    a = ReflexAgent()
    b = ReflexAgent()
    summary = run_tournament(a, b, games=2)
    assert summary.games == 2
    assert summary.draws + sum(summary.wins.values()) == 2
    for data in summary.timing.values():
        assert "avg_time" in data
