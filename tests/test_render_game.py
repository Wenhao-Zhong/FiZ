"""Portable diagram rendering validates histories and derives outcomes."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("render_game", Path(__file__).resolve().parents[1] / "scripts/render_game.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


def test_generic_game_derives_winning_line_and_renders_numbers(tmp_path):
    record = {"history": [0, 15, 1, 16, 2, 17, 3, 18, 4], "model_color": 1}
    history, color, result, line = renderer.validated_result(record)
    assert result == "MODEL WIN" and line == [0, 1, 2, 3, 4]
    paths = renderer.render_game(record, tmp_path)
    assert len(paths) == 2 and all(path.stat().st_size > 10000 for path in paths)
    svg = (tmp_path / "board.svg").read_text("utf-8")
    assert all(f"<!-- {i} -->" in svg for i in range(1, 10))
    assert "Black: FiZ | White: Rapfi" in svg
    assert "Numbers are move order" not in svg and "MODEL WIN" not in svg


def test_incomplete_game_stays_incomplete_and_contradictory_score_rejected():
    record = {"history": [112, 113], "model_color": -1}
    assert renderer.validated_result(record)[2] == "INCOMPLETE"
    with pytest.raises(ValueError, match="terminal"):
        renderer.validated_result(dict(record, complete=True))
    with pytest.raises(ValueError, match="score"):
        renderer.validated_result(dict(record, score=.5))


def test_invalid_history_rejected_before_any_output(tmp_path):
    with pytest.raises(ValueError):
        renderer.render_game({"history": [0, 0]}, tmp_path / "bad")
    with pytest.raises(ValueError, match="integer"):
        renderer.render_game({"history": [True]}, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_arena_color_field_preserves_white_player_result():
    record = {"history": [0, 15, 1, 16, 2, 17, 3, 18, 4], "color": -1, "score": 0.}
    assert renderer.validated_result(record)[1:3] == (-1, "OPPONENT WIN")


def test_figure_header_uses_nominal_checkpoint_and_only_player_names(tmp_path):
    record = {"history": [0, 15, 1, 16, 2, 17, 3, 18, 4], "model_color": -1,
              "slot_hour": 20, "training_hours": 20.3083, "simulations": 2048,
              "missed_immediate_wins": 2, "missed_unique_blocks": 3}
    renderer.render_game(record, tmp_path)
    svg = (tmp_path / "board.svg").read_text("utf-8")
    assert "Checkpoint 20h" in svg and "Black: Rapfi | White: FiZ" in svg
    assert all(text not in svg for text in ("actual", "20.31", "PUCT", "Missed", "OPPONENT WIN", "freestyle Gomoku"))
