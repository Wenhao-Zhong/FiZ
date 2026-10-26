"""Regression tests for the audited search failures, without a neural teacher.

The fake evaluators isolate value scale, visit allocation and bookkeeping. They
are never used as self-play targets or as claims about learned playing strength.
"""
import numpy as np
import pytest

from az_gomoku.search import native


def finish(forest, value=None, prior=None, leaves=4):
    calls = 0
    while not forest.done():
        ids, boards, colors, _ = forest.request(leaves)
        if ids:
            probs = (boards == 0).astype(np.float32)
            if prior is not None:
                probs = np.asarray(prior(boards, colors), dtype=np.float32)
                probs[boards != 0] = 0
                empty = probs.sum(-1) == 0
                probs[empty] = (boards[empty] == 0).astype(np.float32)
            probs /= probs.sum(-1, keepdims=True)
            values = np.zeros(len(ids), np.float32) if value is None else np.asarray(value(boards, colors), dtype=np.float32)
            forest.complete(ids, probs, values)
        calls += 1
        assert calls < 5000, "Search must make progress and respect its visit cap"
    return np.asarray(forest.policy(1.0)), forest.root_stats()


@pytest.mark.parametrize("leaves", [1, 4])
def test_parent_fpu_prevents_losing_side_single_visit_fanout(leaves):
    forest = native().Forest(2, 101)
    forest.set_history(1, [112])
    forest.begin(128, 1.5, 0.0, 0.03)
    policy, stats = finish(forest, value=lambda boards, colors: np.asarray(colors) * 0.99, leaves=leaves)
    # The old Q=0 search visited 127 distinct white moves once, at both batch
    # sizes. With parent-scaled FPU both sides actually investigate continuations.
    for row in stats:
        assert row["visits"] == 128
        assert row["visited_children"] < 127
        assert row["max_visit_ties"] < 127
        assert row["max_depth"] > 1
    np.testing.assert_allclose(policy.sum(-1), 1.0, atol=1e-6)
    assert stats[0]["network_value"] == pytest.approx(0.99)
    assert stats[1]["network_value"] == pytest.approx(-0.99)


def test_exact_puct_ties_are_uniform_even_for_adjacent_candidates():
    # A cyclic random start picked action 0 with probability 224/225 when the
    # only tied best actions were 0 and 1. Reservoir sampling must not do that.
    count = 512
    forest = native().Forest(count, 9321)
    forest.begin(2, 1.5, 0.0, 0.03)

    def two_choices(boards, colors):
        result = np.zeros_like(boards, dtype=np.float32)
        result[:, :2] = 0.5
        return result

    policy, _ = finish(forest, prior=two_choices, leaves=1)
    choices = policy.argmax(-1)
    assert np.isin(choices, [0, 1]).all()
    assert 0.40 < np.mean(choices == 0) < 0.60


def test_mixed_full_and_cheap_caps_and_pruning_are_per_tree():
    forest = native().Forest(4, 15)
    budgets = [128, 512, 128, 512]
    exploratory = [False, True, False, True]
    forest.begin(512, 1.5, 0.25, 0.03, 0.2, 0.2, 2.0, budgets, exploratory)
    raw, stats = finish(forest)
    assert [row["visits"] for row in stats] == budgets
    assert forest.counters()["simulations"] == sum(budgets)
    target = np.asarray(forest.policy(1.0, True))
    np.testing.assert_array_equal(target[[0, 2]], raw[[0, 2]])
    np.testing.assert_allclose(target.sum(-1), 1.0, atol=1e-6)
    assert np.isfinite(target).all() and (target >= 0).all()


def test_retained_subtree_counts_toward_next_moves_total_cap():
    forest = native().Forest(1, 7)
    forest.begin(128, 1.5, 0.0, 0.03)
    raw, _ = finish(forest, value=lambda boards, colors: np.asarray(colors) * 0.99)
    action = int(raw[0].argmax())
    forest.advance([action])
    retained = forest.root_stats()[0]["visits"]
    assert retained > 0
    before = forest.counters()["simulations"]
    forest.begin(128, 1.5, 0.0, 0.03)
    _, stats = finish(forest, value=lambda boards, colors: np.asarray(colors) * 0.99)
    assert stats[0]["visits"] == 128
    assert forest.counters()["simulations"] - before == 128 - retained
    # Repeating an equal or smaller cap performs no extra work.
    before = dict(forest.counters())
    forest.begin(64, 1.5, 0.0, 0.03)
    assert forest.done()
    assert dict(forest.counters()) == before
    assert forest.root_stats()[0]["new_simulations"] == 0


def test_target_pruning_reduces_exploration_without_mutating_action_counts():
    forest = native().Forest(1, 444)
    forest.begin(512, 1.5, 0.1, 1.0, 0.2, 0.2, 2.0, [], [True])

    def prior(boards, colors):
        p = np.zeros_like(boards, dtype=np.float32)
        p[:, 0], p[:, 1] = 0.7, 0.3
        return p

    def value(boards, colors):
        black_wins = np.where(boards[:, 0] == 1, 0.9, -0.9)
        return black_wins * np.asarray(colors)

    raw, stats = finish(forest, value=value, prior=prior)
    greedy = np.asarray(forest.policy(0.0))
    target = np.asarray(forest.policy(1.0, True))
    assert stats[0]["forced_playouts"] > 0
    assert not np.array_equal(raw, target)
    assert target[0, raw[0].argmax()] > 0
    np.testing.assert_allclose(target.sum(-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(forest.policy(1.0), raw)
    np.testing.assert_array_equal(forest.policy(0.0), greedy)


def test_pruning_keeps_all_tied_maxima_and_one_visit_root_fallback():
    forest = native().Forest(1, 9)
    forest.begin(1, 1.5, 0.25, 0.03, 0.2, 0.2, 2.0)
    finish(forest)
    np.testing.assert_allclose(np.asarray(forest.policy(1.0, True)).sum(-1), 1.0, atol=1e-6)
    forest.clear_trees()
    forest.begin(16, 1.5, 0.0, 0.03, 0.2, 0.0, 2.0, [], [True])
    raw, stats = finish(forest)
    assert stats[0]["max_visit_ties"] == 15
    np.testing.assert_array_equal(forest.policy(1.0, True), raw)
    # Noisy root with no forced allowance is exactly the unpruned policy.
    forest.clear_trees()
    forest.begin(64, 1.5, 0.25, 0.03, 0.2, 0.2, 0.0)
    raw, _ = finish(forest)
    np.testing.assert_array_equal(forest.policy(1.0, True), raw)


@pytest.mark.parametrize("color", [1, -1])
def test_terminal_win_sign_both_colors_at_fixed_full_budget(color):
    # Root reduction is zero here to test terminal signs independently of a
    # learned prior. A broad uniform prior is intentionally not a tactical oracle.
    history = [109, 108, 110, 0, 111, 16, 113, 32]
    if color == -1:
        history = [200] + history
    forest = native().Forest(1, 109)
    forest.set_history(0, history)
    forest.begin(512, 1.5, 0.0, 0.03, 0.2, 0.0)
    raw, _ = finish(forest)
    assert raw[0].argmax() == 112
    forest.advance([112])
    assert forest.states()[5][0]
    assert forest.states()[4][0] == color


def test_rejects_invalid_mixed_caps_and_nonfinite_search_parameters():
    forest = native().Forest(2, 1)
    with pytest.raises(RuntimeError):
        forest.begin(128, 1.5, 0.0, 0.03, budgets=[128])
    with pytest.raises(RuntimeError):
        forest.begin(128, 1.5, 0.0, 0.03, budgets=[0, 128])
    with pytest.raises(RuntimeError):
        forest.begin(128, float("nan"), 0.0, 0.03)
