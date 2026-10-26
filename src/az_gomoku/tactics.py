"""Evaluation-only tactical opponent and checksum-validated tactical suite.

These helpers supply no self-play actions, rewards or training labels.
Colors are 0 (black) and 1 (white); board stones are +1 and -1.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .rules import RULE_ID


LINES = np.asarray([
    [(r + j * dr) * 15 + c + j * dc for j in range(5)]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(15) for c in range(15)
    if 0 <= r + 4 * dr < 15 and 0 <= c + 4 * dc < 15
], dtype=np.int32)


OPPONENT_ID = {"kind": "one_ply_tactical", "version": 1,
               "priority": ["immediate_win", "block_immediate_loss", "uniform_legal"],
               "tie_break": "uniform_locked_game_rng", "training_use": False}


def winning_moves(board, color):
    flat = np.asarray(board).ravel()
    windows = flat[LINES]
    ready = (np.sum(windows == (1 - 2 * color), axis=1) == 4) & (np.sum(windows == 0, axis=1) == 1)
    return np.unique(LINES[ready][windows[ready] == 0])


def choose_tactical(state, rng):
    for color in (state.to_play, 1 - state.to_play):
        moves = winning_moves(state.board, color)
        if len(moves):
            return int(rng.choice(moves))
    return int(rng.choice(np.flatnonzero(state.legal_mask)))


def load_suite(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if actual != value.get("sha256") or value.get("rule_id") != RULE_ID:
        raise ValueError("Invalid tactical suite identity")
    return value
