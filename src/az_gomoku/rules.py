"""Independent reference rules for 15x15 freestyle Gomoku.

Black moves first. Five or more contiguous stones win; there are no forbidden
moves, passes or swap openings. The native search engine is tested against this
small implementation.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

BOARD_SIZE = 15
NUM_ACTIONS = BOARD_SIZE ** 2
RULE_ID = "Freestyle-Gomoku15-v1"


def readonly(value, dtype=None):
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class GomokuState:
    board: np.ndarray
    to_play: int = 0
    history: tuple[int, ...] = ()
    winner: int = -1
    terminated: bool = False

    @cached_property
    def legal_mask(self):
        return readonly((self.board.ravel() == 0) & ~np.bool_(self.terminated), np.bool_)


def reference_replay(history):
    """Independent Python rules, contiguous rays rather than five-cell windows."""
    board = [[0] * 15 for _ in range(15)]
    winner, done = -1, False
    for move, action in enumerate(history):
        if done or not isinstance(action, (int, np.integer)) or not 0 <= action < 225:
            raise ValueError("Invalid reference history")
        r, c = divmod(int(action), 15)
        if board[r][c]:
            raise ValueError("Occupied reference point")
        stone = 1 if move % 2 == 0 else -1
        board[r][c] = stone
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            length = 1
            for sign in (-1, 1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < 15 and 0 <= cc < 15 and board[rr][cc] == stone:
                    length += 1
                    rr += sign * dr
                    cc += sign * dc
            if length >= 5:
                winner, done = move % 2, True
        done = done or move == 224
    return GomokuState(readonly(board, np.int8), len(history) % 2, tuple(map(int, history)), winner, done)
