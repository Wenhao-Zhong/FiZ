"""Independent rule checks, search value signs, real optimizer updates and resume."""
from dataclasses import asdict
import shutil

import numpy as np
import pytest
import torch

from az_gomoku.model import BoardTransformer, NetworkConfig, make_planes, infer
from az_gomoku.search import native
from az_gomoku.train import Config, Replay, Session, augment, run_search
from az_gomoku.rules import reference_replay


@pytest.mark.parametrize("direction", [(0, 1), (1, 0), (1, 1), (1, -1)])
@pytest.mark.parametrize("color", [0, 1])
def test_native_four_directions_and_both_colors(direction, color):
    dr, dc = direction
    line = [(2+i*dr)*15+7+i*dc for i in range(5)]
    filler = [210, 212, 214, 216, 218]
    history = [v for pair in zip(line[:4], filler[:4]) for v in pair] + [line[-1]] if color == 0 else [v for pair in zip(filler, line) for v in pair]
    forest = native().Forest(1, 1)
    for k, action in enumerate(history):
        forest.advance([action])
        boards, colors, last, ply, winner, done = forest.states()
        ref = reference_replay(history[:k+1])
        np.testing.assert_array_equal(boards[0], ref.board.ravel())
        assert done[0] == ref.terminated
        assert winner[0] == (0 if ref.winner < 0 else 1 if ref.winner == 0 else -1)
    assert done[0]
    with pytest.raises(RuntimeError):
        forest.advance([200])


def test_random_complete_games_overline_draw_and_no_wrap():
    rng = np.random.default_rng(918)
    forest = native().Forest(16, 1)
    histories = [[] for _ in range(16)]
    for step in range(225):
        boards, _, _, _, _, done = forest.states()
        if all(done):
            break
        actions = [-1 if done[i] else int(rng.choice(np.flatnonzero(board == 0))) for i, board in enumerate(boards)]
        forest.advance(actions)
        boards, _, _, _, winners, ended = forest.states()
        for i, a in enumerate(actions):
            if a >= 0:
                histories[i].append(a)
            ref = reference_replay(histories[i])
            assert ended[i] == ref.terminated
            assert winners[i] == (0 if ref.winner < 0 else 1 if ref.winner == 0 else -1)
    overline = [0, 210, 1, 212, 2, 214, 4, 216, 5, 218, 3]
    forest.set_history(0, overline)
    assert forest.states()[4][0] == 1
    forest.set_history(0, [13, 210, 14, 212, 15, 214, 16, 216, 17])
    assert not forest.states()[5][0]
    r, c = np.indices((15, 15))
    board = np.where((r + 2*c) % 4 < 2, 1, -1)
    black, white = np.flatnonzero(board.ravel() == 1), np.flatnonzero(board.ravel() == -1)
    history = [int(x) for pair in zip(black[:112], white) for x in pair] + [int(black[-1])]
    forest.set_history(0, history)
    assert forest.states()[5][0] and forest.states()[4][0] == 0


def fake_search(forest, simulations, predictive=False):
    from az_gomoku.tactics import winning_moves
    # This fixture isolates sign/terminal correctness with uniform value=0.
    # Production FPU behavior is tested separately under saturated predictions.
    forest.begin(simulations, 1.5, 0., .03, root_fpu_reduction=0.)
    requests = 0
    while not forest.done():
        ids, boards, colors, last = forest.request(4)
        assert len(ids) == len(set(ids))
        if ids:
            policy = (boards == 0).astype(np.float32)
            if predictive:
                # Unit-test fixture for a known strong child prior, not training data.
                for i, board in enumerate(boards):
                    wins = winning_moves(board, 0 if colors[i] == 1 else 1)
                    if len(wins):
                        policy[i] = 0
                        policy[i, wins] = 1
            policy /= policy.sum(-1, keepdims=True)
            forest.complete(ids, policy, np.zeros(len(ids), np.float32))
            requests += len(ids)
    return forest.policy(0.), requests


def test_search_terminal_sign_mask_and_batched_virtual_losses():
    # Black has a unique gap-fill win at row7,col7.
    history = [109, 108, 110, 0, 111, 16, 113, 32]
    forest = native().Forest(2, 109)
    forest.set_history(0, history)
    forest.set_history(1, history)
    policy, requests = fake_search(forest, 512)
    assert np.argmax(policy[0]) == 112 and np.argmax(policy[1]) == 112
    assert requests < 1024  # terminal leaves are backed up without network calls
    np.testing.assert_allclose(policy.sum(-1), 1)
    assert policy[:, 109].sum() == 0
    assert forest.counters()["simulations"] == 1024
    # Opponent has a unique immediate win. A prior that recognizes own wins
    # at child nodes makes every nonblock lose, testing the second sign flip.
    forest = native().Forest(1, 110)
    forest.set_history(0, [108, 109, 0, 110, 16, 111, 48, 113])
    policy, _ = fake_search(forest, 1024, predictive=True)
    assert np.argmax(policy[0]) == 112


def test_d4_features_policy_and_noncausal_attention_gradient():
    torch.manual_seed(12)
    cfg = NetworkConfig(width=32, heads=2, layers=1, ff=64)
    model = BoardTransformer(cfg)
    boards = np.zeros((8,225), np.int8)
    boards[:, 19] = 1
    planes = make_planes(boards, np.ones(8), np.full(8,19))
    pi = torch.zeros((8,225)); pi[:,19] = 1
    x,p = augment(planes, pi, np.arange(8))
    for i in range(8):
        assert torch.equal(x[i,0].flatten(), p[i])
        assert torch.equal(x[i,2].flatten(), p[i])
    x.requires_grad_()
    logits, value = model(x)
    (logits[0,0] + value[0]).backward()
    assert x.grad[0,:,14,14].abs().sum() > 0
    assert torch.isfinite(model.blocks[0].attention.log_frequency.grad).all()
    policy, values = infer(model, boards, [1]*8, [19]*8, "cpu", False)
    assert np.isfinite(values).all() and not policy[:,19].any()
    np.testing.assert_allclose(policy.sum(-1), 1, atol=1e-6)


def test_real_updates_and_exact_resume_of_partial_games(tmp_path):
    cfg = Config(network=asdict(NetworkConfig(width=32, heads=2, layers=1, ff=64)),
                 actors=4, simulations=8, leaves_per_actor=2, replay_capacity=512,
                 min_replay=16, batch_size=16, steps_per_wave=1, warmup_steps=1,
                 full_search_probability=1., cheap_simulations=8, train_sample_ratio=8.,
                 actor_publish_steps=2, fork_probability=0., random_opening_probability=0.,
                 device="cpu", amp=False)
    a = Session(tmp_path / "continuous", cfg)
    # Legal terminal games supply actual outcome targets; no synthetic labels.
    for prefix in ([], [200,201]):
        board=np.zeros(225,np.int8); samples=[]
        history=prefix+[0,15,1,16,2,17,3,18,4]
        for index, action in enumerate(history):
            pi=np.zeros(225,np.float32);pi[action]=1
            samples.append((board.copy(), 1 if index%2==0 else -1, history[index-1] if index else -1, pi))
            board[action]=1 if index%2==0 else -1
        assert reference_replay(history).winner == 0
        a.replay.append_game(samples,1)
    initial = {k:v.detach().clone() for k,v in a.model.state_dict().items()}
    for _ in range(3):
        a.run_wave()
    a.save()
    other = tmp_path / "resumed";other.mkdir()
    shutil.copy(a.root / "latest.pt", other / "latest.pt")
    shutil.copy(a.root / "initial.pt", other / "initial.pt")
    b = Session(other, cfg)
    assert b.histories == a.histories and b.wave == a.wave
    for _ in range(3):
        a.run_wave()
        b.run_wave()
    assert a.histories == b.histories
    assert a.counters() == b.counters()
    assert a.steps == b.steps == 6
    assert any(not torch.equal(v,initial[k]) for k,v in a.model.state_dict().items())
    for name, value in a.model.state_dict().items():
        torch.testing.assert_close(value,b.model.state_dict()[name],rtol=0,atol=0)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_arena_and_tactical_evaluation_read_correct_test_schema(tmp_path):
    from az_gomoku.evaluate import arena, tactics
    import time
    cfg=Config(network=asdict(NetworkConfig(width=32, heads=2,layers=1,ff=64)),
               device="cpu",amp=False,eval_simulations=4)
    model=BoardTransformer(NetworkConfig(**cfg.network)).eval()
    result=arena(model,cfg,[[]],"one_ply",search=False,raw_path=tmp_path/'games.jsonl')
    assert result['completed_games']==2 and not result['incomplete']
    assert result['summary']['completed_pairs']==1
    tactical=tactics(model,cfg,False,time.monotonic()+60)
    assert tactical['completed']==400
    for group in tactical['summary'].values():
        assert group['black']['n']==100 and group['white']['n']==100


def test_rapfi_protocol_asymmetric_coordinates_and_colors():
    from az_gomoku.train import ROOT
    from az_gomoku.rapfi import Rapfi
    import os
    paths = [os.environ.get(name) for name in ("RAPFI_EXECUTABLE", "RAPFI_CONFIG")]
    if not all(paths):
        pytest.skip("Set RAPFI_EXECUTABLE and RAPFI_CONFIG for the optional engine test")
    executable, config = [ROOT / path for path in paths]
    assert executable.is_file() and config.is_file()
    for color in (1,-1):
        board=np.zeros(225,np.int8)
        board[4*15+4]=-color
        board[4*15+5:4*15+9]=color
        board[[0,16,32]]= -color
        if color==-1:
            board[200]=1
        with Rapfi(executable,config) as engine:
            assert engine.move(board,color)==4*15+9
