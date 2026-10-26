"""Deterministic, evaluation-only development probes, independent of final tests.

Development positions have 12/13, 16/17 or 20/21 stones.  The original frozen
final tactical generator uses 8/9 stones, so those position sets are disjoint
by construction.  This module never opens a final test file and is not used
by self-play, replay sampling or the optimizer.
"""
from functools import lru_cache
import hashlib
import json

import numpy as np

from .rules import reference_replay
from .tactics import winning_moves


def canonical_board(board):
    board = np.asarray(board, dtype=np.int8).reshape(15, 15)
    return min(np.ascontiguousarray(np.rot90(x, k)).tobytes()
               for x in (board, board[:, ::-1]) for k in range(4))


def _item(rng, kind, color, number, seen):
    direction = number % 4
    dr, dc = ((0, 1), (1, 0), (1, 1), (1, -1))[direction]
    # All development boards differ in stone count from the old final suite.
    plies = (12, 16, 20)[number % 3] + color
    for _ in range(10000):
        r, c = map(int, rng.integers(0, 15, size=2))
        if not (0 <= r + 5 * dr < 15 and 0 <= c + 5 * dc < 15):
            continue
        line = [(r + j * dr) * 15 + c + j * dc for j in range(6)]
        threat_color = color if kind == "immediate_win" else 1 - color
        stones = {threat_color: list(line[1:5]), 1 - threat_color: []}
        if kind != "forced_loss":
            stones[1 - threat_color].append(line[0])
        remaining = [a for a in range(225) if a not in line]
        rng.shuffle(remaining)
        counts = {0: (plies + 1) // 2, 1: plies // 2}
        for side in (0, 1):
            while len(stones[side]) < counts[side]:
                stones[side].append(remaining.pop())
            rng.shuffle(stones[side])
        history = [stones[i % 2][i // 2] for i in range(plies)]
        try:
            state = reference_replay(history)
        except ValueError:
            continue
        if state.terminated or state.to_play != color:
            continue
        own = winning_moves(state.board, color).tolist()
        opponent = winning_moves(state.board, 1 - color).tolist()
        if kind == "immediate_win" and own != [line[5]]:
            continue
        if kind == "forced_block" and (own or opponent != [line[5]]):
            continue
        if kind == "forced_loss" and (own or len(opponent) < 2):
            continue
        key = canonical_board(state.board)
        if key in seen:
            continue
        seen.add(key)
        return {"id": f"dev_{kind}_{color}_{number:04d}", "kind": kind,
                "color": color, "direction": direction, "history": history,
                "correct_actions": [line[5]] if kind != "forced_loss" else [],
                "certified_value": 1 if kind == "immediate_win" else -1 if kind == "forced_loss" else None}
    raise RuntimeError("Unable to construct a unique legal development probe")


@lru_cache(maxsize=8)
def generate_dev_suite(per_stratum=16, seed=910241):
    if per_stratum < 4 or per_stratum % 4:
        raise ValueError("dev_tactics_per_stratum must be positive and divisible by four")
    rng = np.random.default_rng(seed)
    seen, entries, value_cases = set(), [], []
    for kind in ("immediate_win", "forced_block", "forced_loss"):
        for color in (0, 1):
            for number in range(per_stratum):
                item = _item(rng, kind, color, number, seen)
                if kind != "forced_loss":
                    entries.append(item)
                if item["certified_value"] is not None:
                    value_cases.append(item)
    result = {"version": 2, "seed": seed, "per_stratum": per_stratum,
              "use": "development_only_never_training_or_final_test",
              "value_target_source": "rule-certified immediate win (+1), or opponent has at least two distinct immediate wins with no own immediate win (-1); optimal local continuation, not predictions from the model",
              "entries": entries, "value_cases": value_cases}
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result
