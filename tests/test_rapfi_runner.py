import importlib.util
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("public_rapfi_runner",
    Path(__file__).resolve().parents[1] / "scripts/evaluate_rapfi.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_simulation_override_does_not_mutate_training_configuration():
    original = runner.Config()
    before = asdict(original)
    effective = runner.evaluation_config(original, 1024)
    assert asdict(original) == before
    assert {key for key, value in asdict(effective).items() if value != before[key]} == {"eval_simulations"}
    assert effective.eval_simulations == 1024 and effective.simulations == original.simulations == 512


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_simulation_override_rejects_nonpositive_or_noninteger_counts(count):
    with pytest.raises(ValueError, match="positive integer"):
        runner.evaluation_config(runner.Config(), count)


def test_cli_override_is_bound_before_model_loading_and_never_relabels_results(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    config = runner.Config(rapfi_assets=["weights"])
    content = json.dumps(asdict(config))
    config_path.write_text(content, encoding="utf-8")
    args = SimpleNamespace(config=config_path, model=tmp_path / "model.pt", output=tmp_path / "results",
                           hours=2., simulations=1024, raw_demo=False)
    monkeypatch.setattr(runner, "rapfi_manifest", lambda *a: {"executable": "exe", "config": "cfg", "asset_0": "weights"})
    monkeypatch.setattr(runner, "source_manifest", lambda *a: {"source": "hash"})
    monkeypatch.setattr(runner, "file_hash", lambda *a: "hash")
    captured = []
    bind = runner.bind_plan
    class PlanReached(Exception): pass
    def capture(output, identity, hours):
        captured.append(identity)
        raise PlanReached
    monkeypatch.setattr(runner, "bind_plan", capture)
    monkeypatch.setattr(runner.torch, "load", lambda *a, **k: pytest.fail("Model loading must follow plan validation"))
    with pytest.raises(PlanReached): runner.run(args)
    assert captured[0]["simulations"] == captured[0]["config"]["eval_simulations"] == 1024
    assert captured[0]["config"]["simulations"] == 512
    bind(args.output, captured[0], args.hours)
    args.simulations = 512
    with pytest.raises(PlanReached): runner.run(args)
    assert captured[1]["openings"] == captured[0]["openings"]
    assert captured[1]["config"]["eval_seed"] == captured[0]["config"]["eval_seed"]
    with pytest.raises(ValueError, match="different"):
        bind(args.output, captured[1], args.hours)
    assert config_path.read_text("utf-8") == content


def test_two_sets_have_fifty_pairs_and_fixed_four_ply_order():
    bank = {"openings": [{"split": "dev", "actions": [1, 2, 3, 4]},
                         {"split": "test", "actions": [1, 2]}]
            + [{"split": "test", "actions": [i, 100, 101, 102]} for i in range(60)]}
    sets = runner.opening_sets(bank)
    assert list(sets) == ["empty", "four_ply"]
    assert len(sets["empty"]) == len(sets["four_ply"]) == 50
    assert all(x == [] for x in sets["empty"])
    assert [x[0] for x in sets["four_ply"]] == list(range(50))


def test_resume_keeps_deadline_and_refuses_protocol_change(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.time, "time", lambda: 100.)
    original = runner.bind_plan(tmp_path, {"model": "one"}, 2.)
    monkeypatch.setattr(runner.time, "time", lambda: 900.)
    assert runner.bind_plan(tmp_path, {"model": "one"}, 2.) == original
    assert original["deadline_epoch"] == 7300.
    with pytest.raises(ValueError, match="different"):
        runner.bind_plan(tmp_path, {"model": "replacement"}, 2.)
    with pytest.raises(ValueError, match="different"):
        runner.bind_plan(tmp_path, {"model": "one"}, 3.)


def test_fresh_engine_at_game_boundary_and_record_response_time(monkeypatch, tmp_path):
    engines = []
    class Engine:
        transcript = ["mock"]
        def __init__(self): self.closed, self.commands = False, []
        def send(self, command): self.commands.append(command)
        def move(self, board, color): return int(np.flatnonzero(np.asarray(board) == 0)[0])
        def close(self): self.closed = True
    def factory(*args):
        engines.append(Engine())
        return engines[-1]
    monkeypatch.setattr(runner.ev, "_rapfi_adapter", factory)
    wrapper = runner.FreshRapfi("engine", "config", float("inf"), tmp_path, "empty")
    for count in [1, 3, 9, 0, 2]:
        board = np.zeros(225);board[:count] = 1
        wrapper.move(board, -1)
    assert len(engines) == 2 and engines[0].closed and not engines[1].closed
    assert all(e.commands == ["INFO thread_num 1\nINFO pondering 0"] for e in engines)
    wrapper.close()
    assert engines[1].closed
    rows = [json.loads(line) for line in (tmp_path / "rapfi_timing.jsonl").read_text().splitlines()]
    assert [x["game_in_attempt"] for x in rows] == [1, 1, 1, 2, 2]
    assert all(x["response_seconds"] >= 0 for x in rows)


def test_scores_separate_colors_and_ignore_unplayed_games():
    result = runner.scores([{"score": 1., "color": 1}, {"score": 0., "color": -1}])
    assert result["overall"]["games"] == 2 and result["overall"]["score_rate"] == .5
    assert result["black"]["wins"] == 1 and result["white"]["losses"] == 1


def test_batched_router_keeps_engines_per_slot_and_restores_adapter(monkeypatch, tmp_path):
    created = []
    original = runner.ev.opponent_moves
    class Engine:
        def __init__(self, executable, config, deadline, output, group):
            self.group, self.closed, self.calls = group, False, []
            created.append(self)
        def move(self, board, color):
            self.calls.append((int(np.count_nonzero(board)), color))
            return 42
        def close(self): self.closed = True
    class Forest:
        def states(self): return np.eye(4), [1, -1, 1, -1]
    def arena(*args, **kwargs):
        assert kwargs["batch"] == 16
        callback = runner.ev.opponent_moves
        callback("rapfi", Forest(), [0, 3], None, None, None, True)
        callback("rapfi", Forest(), [3], None, None, None, True)
        raise RuntimeError("test interruption")
    monkeypatch.setattr(runner, "FreshRapfi", Engine)
    monkeypatch.setattr(runner.ev, "arena", arena)
    with pytest.raises(RuntimeError, match="test interruption"):
        runner.paired_arena(None, None, [[]], float("inf"), tmp_path, "empty", "engine", "config")
    assert runner.ev.opponent_moves is original
    assert len(created) == 2 and all(engine.closed for engine in created)
    assert [len(engine.calls) for engine in created] == [1, 2]


def test_invalid_raw_resume_is_not_a_finished_game():
    with pytest.raises(ValueError, match="Invalid saved"):
        runner.verify_raw({"history": [], "color": 1, "winner": 0, "score": .5})
    failed = {"incomplete": True, "reason": "original deadline"}
    assert runner.verify_raw(failed) == failed
