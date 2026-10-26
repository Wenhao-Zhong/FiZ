"""Evaluation tests protect held-out data, RNG independence and incomplete results."""
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from az_gomoku import evaluate as ev
from az_gomoku.dev_suite import canonical_board, generate_dev_suite
from az_gomoku.model import BoardTransformer, NetworkConfig
from az_gomoku.train import Config
from az_gomoku.rules import reference_replay
from az_gomoku.tactics import winning_moves


def tiny_config(**kwargs):
    return Config(network={"size": 15, "width": 16, "layers": 1, "heads": 1, "ff": 32},
                  device="cpu", amp=False, eval_simulations=8, final_pairs=1, empty_pairs=1,
                  rapfi_final_pairs=1, rapfi_executable="external/rapfi/engine",
                  rapfi_config="external/rapfi/config.toml", **kwargs)


def uniform_infer(model, boards, colors, last, device, amp):
    p = (np.asarray(boards) == 0).astype(np.float32)
    return p / p.sum(-1, keepdims=True), np.zeros(len(boards), np.float32)


def test_dev_suite_legal_balanced_unique_and_disjoint_from_final():
    suite = generate_dev_suite(8, 910241)
    assert suite is generate_dev_suite(8, 910241)
    entries = suite["entries"] + [x for x in suite["value_cases"] if x["kind"] == "forced_loss"]
    boards = []
    for item in entries:
        state = reference_replay(item["history"])
        assert not state.terminated and state.to_play == item["color"]
        assert len(item["history"]) >= 12
        own = winning_moves(state.board, item["color"]).tolist()
        opponent = winning_moves(state.board, 1-item["color"]).tolist()
        if item["kind"] == "immediate_win":
            assert own == item["correct_actions"] and item["certified_value"] == 1
        elif item["kind"] == "forced_block":
            assert not own and opponent == item["correct_actions"] and item["certified_value"] is None
        else:
            assert not own and len(opponent) >= 2 and item["certified_value"] == -1
        boards.append(canonical_board(state.board))
    assert len(set(boards)) == len(boards)
    for kind in ("immediate_win", "forced_block"):
        for color in (0, 1):
            chosen = [x for x in suite["entries"] if x["kind"] == kind and x["color"] == color]
            assert len(chosen) == 8
            assert {d: sum(x["direction"] == d for x in chosen) for d in range(4)} == dict.fromkeys(range(4), 2)
    # Identity audit only: runtime development never opens this final test file.
    final = json.loads((Path(__file__).resolve().parents[1]/"data/gomoku_tactics.json").read_text("utf-8"))
    final_boards = {canonical_board(reference_replay(x["history"]).board) for x in final["entries"]}
    assert not set(boards) & final_boards


def test_paired_interval_boundary_and_complete_pairs_only():
    games = [{"pair": i, "color": c, "score": 1.} for i in range(24) for c in (1, -1)]
    stats = ev.paired_interval(games)
    assert stats["bootstrap_ci95"] == [1., 1.]
    assert 0 < stats["ci95"][0] < 1 and stats["ci95"][1] == 1
    assert "Hoeffding" in stats["ci_method"]
    extra = ev.paired_interval(games + [{"pair": 25, "color": 1, "score": 0.}])
    assert extra["mean"] == extra["black"] == extra["white"] == 1
    assert extra["unpaired_games_excluded"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        ev.paired_interval(games + [games[0]])
    delta = ev.improvement(games, [dict(g, score=0.) for g in games])
    assert delta["mean"] == 1. and delta["ci95"][0] < 1


def test_arena_rng_is_independent_of_batch_and_resume(monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "infer", uniform_infer)
    cfg = tiny_config()
    # An existing unique one-move win makes the tactical opponent end games quickly.
    openings = [[109, 108, 110, 0, 111, 16, 113, 32], [20, 19, 21, 210, 22, 212, 24, 214]]
    a = ev.arena(None, cfg, openings, "one_ply", search=False, batch=1)
    path = tmp_path/"games.jsonl"
    b = ev.arena(None, cfg, openings, "one_ply", search=False, batch=4, raw_path=path)
    keyed = lambda x: {(g["pair"], g["color"]): g["history"] for g in x["games"]}
    assert keyed(a) == keyed(b)
    assert not a["incomplete"] and a["completed_games"] == 4
    before = path.read_bytes()
    resumed = ev.arena(None, cfg, openings, "one_ply", search=False, deadline=-float("inf"), raw_path=path)
    assert keyed(resumed) == keyed(b) and not resumed["incomplete"]
    assert path.read_bytes() == before


def test_deadline_and_rapfi_error_never_become_draws(monkeypatch):
    cfg = tiny_config()
    result = ev.arena(None, cfg, [[]], "one_ply", search=False, deadline=-float("inf"))
    assert result["incomplete"] and result["completed_games"] == 0 and not result["games"]
    def slow_infer(*args):
        time.sleep(.025)
        return uniform_infer(*args)
    monkeypatch.setattr(ev, "infer", slow_infer)
    result = ev.arena(None, cfg, [[]], "random", search=False, deadline=time.monotonic()+.01)
    assert result["incomplete"] and not result["games"]
    monkeypatch.setattr(ev, "infer", uniform_infer)
    class BrokenRapfi:
        def move(self, *args):
            raise TimeoutError("engine test timeout")
    result = ev.arena(None, cfg, [[]], "rapfi", search=False, rapfi=BrokenRapfi())
    assert result["incomplete"] and not result["games"] and "Rapfi error" in result["error"]


def test_search_rng_independent_of_inference_batch(monkeypatch):
    monkeypatch.setattr(ev, "infer", uniform_infer)
    cfg = tiny_config()
    histories = [[0], [112]]
    batched = ev.moves(None, histories, cfg, [np.random.default_rng(3), np.random.default_rng(7)], search=True)
    independent = [ev.moves(None, [h], cfg, np.random.default_rng(seed), search=True)[0]
                   for h, seed in zip(histories, (3, 7))]
    assert batched == independent


def test_search_checks_deadline_after_each_inference(monkeypatch):
    clock = [0.]
    calls = []
    monkeypatch.setattr(ev, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    def expired_infer(*args):
        calls.append(1)
        clock[0] = 2.
        return uniform_infer(*args)
    monkeypatch.setattr(ev, "infer", expired_infer)
    with pytest.raises(ev.Deadline):
        ev.moves(None, [[]], tiny_config(), np.random.default_rng(3), search=True, deadline=1.)
    assert len(calls) == 1


def test_development_does_not_open_final_tactics(monkeypatch):
    read_text = Path.read_text
    def guarded_read(path, *args, **kwargs):
        if path.name == "gomoku_tactics.json":
            raise AssertionError("Development attempted to open final tactics")
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    monkeypatch.setattr(ev, "infer", uniform_infer)
    def probe_result(model, cfg, items, search, deadline, **kwargs):
        return {"requested": len(items), "completed": len(items), "incomplete": False,
                "summary": {k: {c: {"accuracy": 0., "n": 4} for c in ("black", "white")}
                            for k in ("immediate_win", "forced_block")}, "records": []}
    monkeypatch.setattr(ev, "evaluate_tactics", probe_result)
    monkeypatch.setattr(ev, "arena", lambda *a, **k: {"games": [], "incomplete": False,
        "completed_games": 0, "requested_games": 0, "summary": None})
    result = ev.dev_evaluate(None, tiny_config(dev_tactics_per_stratum=4))
    assert result["certified_value_diagnostics"]["completed"] == 16
    assert "rule-certified" in result["certified_value_diagnostics"]["target_source"]


def test_rule_certified_value_diagnostics_are_not_model_labels():
    rows = [{"color": c, "raw_value": -1., "target_value": z,
             "raw_policy_entropy": 4., "raw_policy_top1": .02} for c in (1, -1) for z in (1., -1.)]
    stats = ev._diagnostic_summary(rows, "rule-certified tactical values")
    for color in ("black", "white"):
        assert stats["by_color"][color]["confident_wrong_fraction"] == .5
        assert stats["by_color"][color]["value_mse"] == 2.
        assert stats["by_color"][color]["brier_score"] == .5


def _mock_final_inputs(tmp_path, monkeypatch, *, rapfi_available=True):
    project = tmp_path/"project"; run = tmp_path/"run"; run.mkdir()
    (project/"data").mkdir(parents=True)
    (project/"data/gomoku_openings.json").write_text(json.dumps({"openings": [{"split": "test", "actions": []}]}), encoding="utf-8")
    if rapfi_available:
        exe = project/"external/rapfi/engine"; exe.parent.mkdir(parents=True); exe.touch()
        conf = project/"external/rapfi/config.toml"; conf.parent.mkdir(parents=True, exist_ok=True); conf.touch()
    monkeypatch.setattr(ev, "ROOT", project)
    cfg = tiny_config()
    model = BoardTransformer(NetworkConfig(**cfg.network))
    checkpoint = {"model": model.state_dict(), "elapsed_seconds": 1., "plies": 64, "games": 1, "steps": 1, "search": {}}
    torch.save(checkpoint, run/"latest.pt"); torch.save({"model": model.state_dict()}, run/"initial.pt")
    def fake_tactics(*args):
        return {"requested": 400, "completed": 400, "incomplete": False,
            "summary": {k: {c: {"n": 100, "accuracy": 1.} for c in ("black", "white")} for k in ("immediate_win", "forced_block")}}
    def fake_arena(model, cfg, openings, kind, **kwargs):
        if kwargs.get("deadline", float("inf")) <= time.monotonic():
            return ev._arena_result([], 2*len(openings), "deadline")
        games = [{"pair": i, "color": c, "score": 1.} for i in range(len(openings)) for c in (1, -1)]
        return ev._arena_result(games, 2*len(openings))
    monkeypatch.setattr(ev, "tactics", fake_tactics); monkeypatch.setattr(ev, "arena", fake_arena)
    return run, cfg


def test_final_plan_has_all_rapfi_and_random_baselines(monkeypatch, tmp_path):
    run, cfg = _mock_final_inputs(tmp_path, monkeypatch)
    engines = []
    class Engine:
        transcript = ["mock protocol"]
        def __init__(self): self.closed = False
        def close(self): self.closed = True
    def factory(*args):
        engine = Engine(); engines.append(engine); return engine
    monkeypatch.setattr(ev, "_rapfi_adapter", factory)
    result = ev.final_evaluate(run, cfg, time.monotonic()+60)
    assert len(engines) == 4 and all(x.closed for x in engines)
    for mode in ("raw", "mcts"):
        for name in ("initial", "final"):
            assert not result["comparisons"][f"{mode}_{name}_vs_rapfi_100ms"]["incomplete"]
            assert f"{mode}_{name}_vs_random" in result["comparisons"]
        assert f"{mode}_rapfi_100ms_improvement" in result
    assert not result["incomplete"] and result["basic_tactics_gate_passed"]
    assert not result["single_seed_h1_confirmed"] and not result["professional_strength_established"]


def test_missing_rapfi_invalidates_complete_assessment(monkeypatch, tmp_path):
    run, cfg = _mock_final_inputs(tmp_path, monkeypatch, rapfi_available=False)
    result = ev.final_evaluate(run, cfg, time.monotonic()+60)
    assert result["incomplete"] and not result["single_seed_learning_evidence"]
    missing = [v for k, v in result["comparisons"].items() if "rapfi" in k]
    assert len(missing) == 4 and all(x["incomplete"] and x["completed_games"] == 0 for x in missing)


def test_failed_rapfi_initialization_cleans_child(monkeypatch):
    from az_gomoku import rapfi as module
    class Process:
        killed = False
        def poll(self): return None
        def kill(self): self.killed = True
        def wait(self, timeout): return 1
    process = Process()
    class Broken:
        def __init__(self, *args, **kwargs):
            self.proc = process
            raise TimeoutError("bad handshake")
    monkeypatch.setattr(module, "Rapfi", Broken)
    with pytest.raises(TimeoutError):
        ev._rapfi_adapter("unused", "unused", time.monotonic()+10)
    assert process.killed


def test_tactics_resume_only_missing_ids_and_keep_item_rng(monkeypatch):
    items = generate_dev_suite(4, 910241)["entries"]
    by_history = {tuple(x["history"]): x for x in items}
    calls, random_draws = [], {}
    def probe_moves(model, histories, cfg, rngs, search, deadline):
        calls.extend(tuple(history) for history in histories)
        for history, rng in zip(histories, rngs):
            random_draws[tuple(history)] = int(rng.integers(0, 2**31))
        return [by_history[tuple(history)]["correct_actions"][0] for history in histories]
    monkeypatch.setattr(ev, "moves", probe_moves)
    expected = ev.evaluate_tactics(None, tiny_config(), items, False, float("inf"))
    expected_draws = dict(random_draws)
    prior = ev._tactical_result(items, expected["records"][:5])
    calls.clear(); random_draws.clear(); progress = []
    resumed = ev.evaluate_tactics(None, tiny_config(), items, False, float("inf"), prior=prior, on_progress=progress.append)
    assert resumed == expected
    assert len(calls) == len(items)-5
    assert random_draws == {tuple(item["history"]): expected_draws[tuple(item["history"])] for item in items[5:]}
    assert progress[-1]["completed"] == len(items)
    monkeypatch.setattr(ev, "moves", lambda *a, **k: pytest.fail("Completed tactics must not run inference"))
    assert ev.evaluate_tactics(None, tiny_config(), items, False, -float("inf"), prior=expected) == expected
    bad = dict(prior, records=[dict(prior["records"][0], action=items[0]["history"][0])])
    bad["completed"] = 1
    with pytest.raises(ValueError, match="illegal"):
        ev.evaluate_tactics(None, tiny_config(), items, False, float("inf"), prior=bad)


def test_final_resume_preserves_complete_tactics_and_original_start(monkeypatch, tmp_path):
    original_tactics = ev.tactics
    run, cfg = _mock_final_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(ev, "tactics", original_tactics)
    from az_gomoku import tactics as tactics_module
    suite = generate_dev_suite(4, 910241)
    monkeypatch.setattr(tactics_module, "load_suite", lambda path: suite)
    items = {tuple(x["history"]): x for x in suite["entries"]}
    queried = []
    def probe_moves(model, histories, cfg, rngs, search, deadline):
        queried.extend(histories)
        return [items[tuple(history)]["correct_actions"][0] for history in histories]
    monkeypatch.setattr(ev, "moves", probe_moves)
    class Engine:
        transcript = []
        def close(self): pass
    monkeypatch.setattr(ev, "_rapfi_adapter", lambda *args: Engine())
    first = ev.final_evaluate(run, cfg, time.monotonic()+60)
    assert len(queried) == 4*len(suite["entries"])
    monkeypatch.setattr(ev, "moves", lambda *a, **k: pytest.fail("Completed tactical groups must be reused"))
    resumed = ev.final_evaluate(run, cfg, -float("inf"))
    assert resumed["started_at"] == first["started_at"] and len(resumed["resumed_at"]) == 1
    assert resumed["tactics"] == first["tactics"]
    assert resumed["incomplete"]  # mock game records were not persisted; no invented results


def test_final_identity_binds_initial_and_refuses_orphan_game_files(monkeypatch, tmp_path):
    run, cfg = _mock_final_inputs(tmp_path, monkeypatch)
    class Engine:
        transcript = []
        def close(self): pass
    monkeypatch.setattr(ev, "_rapfi_adapter", lambda *args: Engine())
    first = ev.final_evaluate(run, cfg, time.monotonic()+60)
    assert "initial_checkpoint_sha256" in first["identity"]
    initial_bytes = (run/"initial.pt").read_bytes()
    (run/"initial.pt").write_bytes(initial_bytes + b"changed")
    with pytest.raises(RuntimeError, match="different checkpoint"):
        ev.final_evaluate(run, cfg, time.monotonic()+60)
    (run/"initial.pt").write_bytes(initial_bytes)
    (run/"assessment.partial.json").unlink()
    (run/"raw_final_vs_random.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="without its identity manifest"):
        ev.final_evaluate(run, cfg, time.monotonic()+60)


def test_rapfi_engine_log_appends_attempts(monkeypatch, tmp_path):
    run, cfg = _mock_final_inputs(tmp_path, monkeypatch)
    engines = []
    class Engine:
        def __init__(self, number): self.transcript = [f"protocol-attempt-{number}"]
        def close(self): pass
    def factory(*args):
        engine = Engine(len(engines)); engines.append(engine); return engine
    monkeypatch.setattr(ev, "_rapfi_adapter", factory)
    ev.final_evaluate(run, cfg, time.monotonic()+60)
    ev.final_evaluate(run, cfg, time.monotonic()+60)
    transcript = (run/"raw_initial_vs_rapfi_100ms.engine.log").read_text("utf-8")
    assert transcript.count("# Evaluation attempt") == 2
    assert "protocol-attempt-0" in transcript and "protocol-attempt-4" in transcript
