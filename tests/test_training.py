"""Regression checks for new-data accounting and stable self-play actors."""
from dataclasses import asdict, replace
import json

import numpy as np
import pytest
import torch

from az_gomoku.model import NetworkConfig
from az_gomoku.train import Config, Session, move_temperature, tempered_policy


def small_config(**kwargs):
    config = Config(network=asdict(NetworkConfig(width=32, heads=2, layers=1, ff=64, value_head="wdl")),
                    actors=4, simulations=8, cheap_simulations=4, leaves_per_actor=2,
                    replay_capacity=128, min_replay=4, batch_size=8, steps_per_wave=5,
                    train_sample_ratio=2., actor_publish_steps=2, warmup_steps=1,
                    fork_probability=0., random_opening_probability=0., device="cpu", amp=False)
    return replace(config, **kwargs)


def add_four_samples(session):
    board = np.zeros(225, np.int8)
    policy = np.zeros(225, np.float32)
    policy[112] = 1
    session.replay.append_game([(board.copy(), 1, -1, policy.copy()) for _ in range(4)], 1)


def test_only_new_retained_positions_buy_updates_and_actor_publishes_in_chunks(tmp_path):
    session = Session(tmp_path / "credit", small_config())
    original_actor = {k: v.clone() for k, v in session.actor.state_dict().items()}
    add_four_samples(session)
    session.run_wave()
    assert session.steps == 1 and session.training_positions == 4
    assert session.training_credit == 0
    assert all(torch.equal(v, original_actor[k]) for k, v in session.actor.state_dict().items())
    assert any(not torch.equal(v, original_actor[k]) for k, v in session.model.state_dict().items())
    session.run_wave()
    assert session.steps == 1  # replaying existing data does not mint new credit
    add_four_samples(session)
    session.run_wave()
    assert session.steps == 2 and session.actor_step == 2 and session.actor_version == 1
    assert all(torch.equal(v, session.model.state_dict()[k]) for k, v in session.actor.state_dict().items())
    assert all(x["visits"] == 0 for x in session.forest.root_stats())


def test_cheap_moves_not_written_as_policy_training_targets(tmp_path, monkeypatch):
    import az_gomoku.train as training
    original = training.run_search
    captured = []
    def observe(*args, **kwargs):
        captured.append(np.asarray(kwargs["exploratory"]).copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(training, "run_search", observe)
    session = Session(tmp_path / "mixed", small_config(actors=16))
    session.run_wave()
    full = captured[0]
    assert full.any() and (~full).any()
    assert [len(x) for x in session.samples] == full.astype(int).tolist()
    for samples in session.samples:
        for board, color, last, target in samples:
            assert np.isfinite(target).all() and target.sum() == pytest.approx(1., abs=.002)
            assert not target[board != 0].any()
    assert session.plies == 16


def test_temperature_decays_and_cheap_moves_are_greedy():
    cfg = Config()
    assert [move_temperature(p, cfg) for p in (0,4,8,12,30)] == [1., .5, .25, 0., 0.]
    assert move_temperature(0, cfg, False) == 0
    np.testing.assert_allclose(tempered_policy([.4,.4,.2,0],0), [.5,.5,0,0])
    p = tempered_policy([.4,.4,.2,0],.25)
    assert p[2] < .04 and p[3] == 0


def test_credit_capacity_cannot_silently_disable_learning():
    with pytest.raises(ValueError, match="at least one batch"):
        small_config(train_credit_max_samples=7).validate()


def test_prefix_moves_are_counted_but_reused_fork_history_is_not(tmp_path):
    session = Session(tmp_path / "openings", small_config(random_opening_probability=1.))
    session.reset_game(0)
    prefix = list(session.histories[0])
    assert 1 <= len(prefix) <= 6 and session.plies == len(prefix)
    assert session.opening_lengths[0] == len(prefix)
    session.cfg.fork_probability = 1.
    session.cfg.random_opening_probability = 0.
    session.forks = [[0, 16, 32, 48]]
    before = session.plies
    session.reset_game(0)
    assert session.histories[0] == [0,16,32,48]
    assert session.opening_kinds[0] == "selfplay_fork"
    assert session.plies == before


def test_terminated_root_is_rejected_before_mutation(tmp_path):
    root = tmp_path / "terminated"
    root.mkdir()
    (root / "termination.json").write_text('{"reason":"user"}')
    before = list(root.iterdir())
    with pytest.raises(RuntimeError, match="terminated"):
        Session(root, small_config())
    assert list(root.iterdir()) == before


def test_v1_plan_is_rejected_before_lock_or_budget(tmp_path):
    root = tmp_path / "v1"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps({"config":{},"sources":{}}))
    with pytest.raises(RuntimeError, match="v2 requires"):
        Session(root, small_config())
    assert not (root / "process.json").exists() and not (root / "budget.json").exists()


def test_supervisor_freezes_run_local_config_without_overwriting(tmp_path):
    import importlib.util
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / "scripts/run_az_gomoku.py"
    spec = importlib.util.spec_from_file_location("az_supervisor_test", script)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    cfg = small_config()
    path = supervisor.freeze_config(tmp_path, cfg)
    original = path.read_bytes()
    assert json.loads(original) == asdict(cfg)
    assert supervisor.freeze_config(tmp_path, cfg) == path
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        supervisor.freeze_config(tmp_path, replace(cfg, seed=102))
    assert path.read_bytes() == original
