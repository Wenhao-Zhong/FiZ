"""Evaluate FiZ against Rapfi on 100 empty-board and 100 opening games."""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from az_gomoku import evaluate as ev
from az_gomoku.assets import file_hash, rapfi_manifest, rapfi_paths
from az_gomoku.model import BoardTransformer, NetworkConfig
from az_gomoku.rules import reference_replay
from az_gomoku.train import Config, atomic_json, source_manifest


def evaluation_config(config, simulations):
    if isinstance(simulations, bool) or not isinstance(simulations, int) or simulations <= 0:
        raise ValueError("Simulation count must be a positive integer")
    effective = replace(config, eval_simulations=simulations)
    effective.validate()
    return effective


def opening_sets(bank):
    fixed = [item["actions"] for item in bank["openings"]
             if item["split"] == "test" and len(item["actions"]) == 4][:50]
    if len(fixed) != 50:
        raise ValueError("The test bank must contain at least 50 four-ply openings")
    return {"empty": [[] for _ in range(50)], "four_ply": fixed}


def bind_plan(output, identity, hours):
    path = output / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text("utf-8"))
        if plan["identity"] != identity or plan["hours"] != hours:
            raise ValueError("Existing results belong to a different model, assets or protocol")
        return plan
    if any(output.iterdir()):
        raise ValueError("Nonempty output has no evaluation plan")
    now = time.time()
    plan = {"identity": identity, "hours": hours, "started_epoch": now,
            "deadline_epoch": now + hours * 3600}
    atomic_json(path, plan)
    return plan


def acquire(output):
    path = output / "process.json"
    if path.exists():
        old = json.loads(path.read_text("utf-8"))
        try:
            if abs(psutil.Process(old["pid"]).create_time() - old["created"]) < .1:
                raise RuntimeError("This evaluation is already running")
        except psutil.NoSuchProcess:
            pass
        path.unlink()
    process = psutil.Process()
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"pid": process.pid, "created": process.create_time()}, stream)


class FreshRapfi:
    """One arena slot; decreasing occupancy marks the start of the next game."""
    def __init__(self, executable, config, deadline, output, group):
        self.executable, self.config, self.deadline = executable, config, deadline
        self.output, self.group = output, group
        self.engine, self.last_count, self.game = None, -1, 0

    def move(self, board, color):
        count = int(np.count_nonzero(board))
        if self.engine is None or count <= self.last_count:
            self.close()
            self.game += 1
            self.engine = ev._rapfi_adapter(self.executable, self.config, self.deadline)
            try:
                self.engine.send("INFO thread_num 1\nINFO pondering 0")
            except BaseException:
                self.close()
                raise
        self.last_count = count
        started = time.perf_counter()
        action = self.engine.move(board, color)
        with (self.output / "rapfi_timing.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"group": self.group, "game_in_attempt": self.game,
                "board_plies": count, "response_seconds": time.perf_counter() - started}) + "\n")
        ev._check_deadline(self.deadline)
        return action

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            try:
                engine.close()
            finally:
                with (self.output / "rapfi.engine.log").open("a", encoding="utf-8") as stream:
                    stream.write(f"\n# {self.group}, game {self.game}, epoch {time.time()}\n")
                    stream.write("\n".join(engine.transcript) + "\n")


def scores(games):
    def group(rows):
        points = [row["score"] for row in rows]
        return {"games": len(points), "wins": points.count(1.), "draws": points.count(.5),
                "losses": points.count(0.), "score_rate": sum(points) / len(points) if points else None}
    return {"overall": group(games), "black": group([g for g in games if g["color"] == 1]),
            "white": group([g for g in games if g["color"] == -1])}


def paired_arena(model, cfg, openings, deadline, output, name, executable, config):
    """Batch neural inference while isolating each game's external engine state."""
    original = ev.opponent_moves
    engines = {}
    def opponent_moves(kind, forest, indices, rngs, initial, cfg, search,
                       rapfi=None, deadline=deadline):
        if kind != "rapfi":
            return original(kind, forest, indices, rngs, initial, cfg, search, rapfi, deadline)
        boards, colors, *_ = forest.states()
        actions = []
        for slot in indices:
            if slot not in engines:
                engines[slot] = FreshRapfi(executable, config, deadline, output, f"{name}.slot-{slot}")
            actions.append(engines[slot].move(boards[slot], colors[slot]))
        return actions
    # The CLI is single-threaded; restore the adapter even after an exception.
    ev.opponent_moves = opponent_moves
    try:
        return ev.arena(model, cfg, openings, "rapfi", search=True, deadline=deadline,
                        raw_path=output / f"{name}.jsonl", batch=16)
    finally:
        ev.opponent_moves = original
        errors = []
        for engine in engines.values():
            try:
                engine.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Failed to close a Rapfi process") from errors[0]


def raw_demo(model, cfg, engine, deadline):
    history, rng = [], np.random.default_rng(cfg.eval_seed)
    while not reference_replay(history).terminated:
        ev._check_deadline(deadline)
        state = reference_replay(history)
        action = (ev.moves(model, [history], cfg, rng, search=False, deadline=deadline)[0]
                  if state.to_play == 0 else engine.move(state.board.ravel(), -1))
        history.append(int(action))
    state = reference_replay(history)
    winner = 0 if state.winner == -1 else 1 if state.winner == 0 else -1
    return {"history": history, "color": 1, "winner": winner,
            "score": .5 if winner == 0 else float(winner == 1), "demonstration_only": True}


def verify_raw(record):
    if record.get("incomplete"):
        return record  # A failed single-game demonstration is never retried.
    state = reference_replay(record["history"])
    winner = 0 if state.winner == -1 else 1 if state.winner == 0 else -1
    score = .5 if winner == 0 else float(winner == 1)
    if not state.terminated or record["color"] != 1 or record["winner"] != winner or record["score"] != score:
        raise ValueError("Invalid saved Raw demonstration")
    return record


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not 0 < args.hours <= 24:
        raise ValueError("Evaluation hours must be positive and at most 24")
    cfg = evaluation_config(Config(**json.loads(args.config.read_text("utf-8"))), args.simulations)
    assets = rapfi_manifest(cfg, ROOT)
    if not cfg.rapfi_assets or any(value is None for value in assets.values()):
        raise ValueError("Configure the Rapfi executable, config and all model assets first")
    sets = opening_sets(json.loads((ROOT / "data/gomoku_openings.json").read_text("utf-8")))
    identity = {"model_sha256": file_hash(args.model), "config": asdict(cfg), "rapfi_assets": assets,
                "sources": source_manifest(cfg), "runner_sha256": file_hash(Path(__file__)),
                "openings": sets, "raw_demo": args.raw_demo, "simulations": cfg.eval_simulations,
                "rapfi_time_limit_ms": 100, "rapfi_threads": 1, "pondering": False,
                "fresh_rapfi_per_game": True, "colors_per_pair": [1, -1], "batch": 16}
    plan = bind_plan(output, identity, args.hours)
    acquire(output)
    deadline = time.monotonic() + max(0., plan["deadline_epoch"] - time.time())
    torch.set_num_threads(4)
    payload = torch.load(args.model, map_location="cpu", weights_only=True)
    if payload["config"] != cfg.network:
        raise ValueError("Model architecture does not match the configuration")
    model = BoardTransformer(NetworkConfig(**payload["config"])).to(cfg.device).eval()
    model.load_state_dict(payload["model"], strict=True)
    executable, config = rapfi_paths(cfg, ROOT)
    result = {"identity": identity, "groups": {}, "raw_demo": None,
              "initial_baseline_assessed": False, "learning_evidence_assessed": False,
              "professional_strength_established": False, "requested_mcts_games": 200}
    for name, openings in sets.items():
        started = time.perf_counter()
        group = paired_arena(model, cfg, openings, deadline, output, name, executable, config)
        group.update(scores=scores(group["games"]), attempt_seconds=time.perf_counter() - started)
        group["unique_histories"] = len({tuple(game["history"]) for game in group["games"]})
        group["sampling_unit"] = "empty-board seed pair" if name == "empty" else "fixed four-ply opening pair"
        result["groups"][name] = group
        atomic_json(output / "assessment.partial.json", result)
    if args.raw_demo:
        raw_path = output / "raw_demo.json"
        if raw_path.exists():
            result["raw_demo"] = verify_raw(json.loads(raw_path.read_text("utf-8")))
        else:
            engine = FreshRapfi(executable, config, deadline, output, "raw")
            atomic_json(raw_path, {"incomplete": True, "reason": "attempt started; no complete record"})
            try:
                result["raw_demo"] = raw_demo(model, cfg, engine, deadline)
                atomic_json(raw_path, result["raw_demo"])
            except Exception as error:
                result["raw_demo"] = {"incomplete": True, "reason": f"{type(error).__name__}: {error}"}
                atomic_json(raw_path, result["raw_demo"])
            finally:
                engine.close()
    result.update(completed_epoch=time.time(), deadline_epoch=plan["deadline_epoch"],
                  incomplete=any(g["incomplete"] for g in result["groups"].values())
                  or bool(args.raw_demo and result["raw_demo"].get("incomplete")))
    atomic_json(output / "assessment.json", result)
    print(json.dumps({name: group["scores"] for name, group in result["groups"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=2.)
    parser.add_argument("--simulations", type=int, default=512, help="PUCT simulations per move; evaluation only")
    parser.add_argument("--raw-demo", action="store_true", help="Also play one raw-policy game as black")
    run(parser.parse_args())
