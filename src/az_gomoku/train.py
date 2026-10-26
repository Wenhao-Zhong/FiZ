"""Single-GPU AlphaZero-style self-play with bounded runtime and atomic resume."""
from __future__ import annotations
import argparse
from collections import deque
import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from .model import BoardTransformer, NetworkConfig, infer, make_planes
from .search import native

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    seed: int = 101
    network: dict = field(default_factory=lambda: asdict(NetworkConfig(value_head="wdl")))
    actors: int = 64
    simulations: int = 512
    cheap_simulations: int = 128
    full_search_probability: float = .25
    leaves_per_actor: int = 4
    cpuct: float = 1.5
    fpu_reduction: float = .2
    root_fpu_reduction: float = .2
    forced_playouts: float = 2.
    policy_target_pruning: bool = True
    dirichlet_alpha: float = .03
    dirichlet_fraction: float = .25
    temperature_moves: int = 12
    temperature_halflife: float = 4.
    temperature_floor: float = .15
    replay_capacity: int = 500000
    min_replay: int = 8192
    batch_size: int = 256
    # Upper burst limit; actual updates are credited by NEW retained positions.
    steps_per_wave: int = 2
    train_sample_ratio: float = 4.
    train_credit_max_samples: int = 32768
    actor_publish_steps: int = 32
    fork_probability: float = .25
    fork_capacity: int = 2048
    random_opening_probability: float = .25
    random_opening_max_moves: int = 6
    learning_rate: float = .0003
    min_lr_ratio: float = .1
    warmup_steps: int = 1000
    schedule_steps: int = 62500
    weight_decay: float = .0001
    max_grad_norm: float = 5.
    amp: bool = True
    runtime_hours: float = 26.
    final_eval_reserve_hours: float = 2.
    checkpoint_seconds: float = 900.
    dev_eval_seconds: float = 3600.
    dev_pairs: int = 24
    dev_tactics_per_stratum: int = 16
    dev_tactics_seed: int = 910241
    dev_eval_max_seconds: float = 600.
    dev_history_pairs: int = 8
    eval_seed: int = 271828
    basic_tactics_threshold: float = .9
    final_pairs: int = 100
    empty_pairs: int = 32
    rapfi_final_pairs: int = 32
    eval_simulations: int = 512
    max_train_steps: int = 500000
    device: str = "cuda"
    rapfi_executable: str = ""
    rapfi_config: str = ""
    rapfi_assets: list[str] = field(default_factory=list)

    def validate(self):
        NetworkConfig(**self.network)
        if self.network["size"] != 15:
            raise ValueError("The native rule engine is frozen to 15x15")
        if min(self.actors, self.simulations, self.leaves_per_actor, self.batch_size,
               self.steps_per_wave, self.replay_capacity, self.min_replay) < 1:
            raise ValueError("Counts must be positive")
        if self.min_replay > self.replay_capacity or not 0 <= self.dirichlet_fraction <= 1:
            raise ValueError("Invalid replay/noise settings")
        if min(self.cheap_simulations, self.actor_publish_steps, self.fork_capacity,
               self.train_credit_max_samples, self.eval_simulations, self.max_train_steps,
               self.schedule_steps, self.dev_pairs, self.dev_history_pairs,
               self.final_pairs, self.empty_pairs, self.rapfi_final_pairs) < 1:
            raise ValueError("Search, training and evaluation counts must be positive")
        if not 0 < self.full_search_probability <= 1 or not 0 < self.train_sample_ratio:
            raise ValueError("Full-search probability and sample ratio must be positive")
        if self.train_credit_max_samples < self.batch_size:
            raise ValueError("Training credit capacity must allow at least one batch")
        if min(self.fpu_reduction, self.root_fpu_reduction, self.forced_playouts) < 0:
            raise ValueError("Search reductions cannot be negative")
        if self.temperature_moves < 0 or self.temperature_halflife <= 0 or not 0 < self.temperature_floor <= 1:
            raise ValueError("Invalid temperature schedule")
        if min(self.fork_probability, self.random_opening_probability) < 0 or self.fork_probability + self.random_opening_probability > 1:
            raise ValueError("Opening probabilities must sum to at most one")
        if not 0 <= self.random_opening_max_moves <= 8:
            raise ValueError("Random prefixes must end before the earliest possible terminal")
        if self.dev_tactics_per_stratum < 4 or self.dev_tactics_per_stratum % 4 or not 0 < self.basic_tactics_threshold <= 1:
            raise ValueError("Invalid development tactics or gate")
        if self.runtime_hours <= self.final_eval_reserve_hours or self.final_eval_reserve_hours < 0:
            raise ValueError("Runtime must leave a positive training window")


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(7):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.02 * 2 ** attempt)


def source_manifest(cfg=None):
    from .assets import file_hash, rapfi_manifest
    files = sorted((ROOT / "src/az_gomoku").glob("*.py"))
    files += sorted((ROOT / "src/az_gomoku").glob("*.cpp"))
    files += [ROOT / "scripts/build_az_search.py", ROOT / "scripts/run_az_gomoku.py"]
    files += [ROOT / "data/gomoku_openings.json", ROOT / "data/gomoku_tactics.json"]
    files += [Path(native().__file__)]
    result = {p.relative_to(ROOT).as_posix(): file_hash(p) for p in files}
    result["external_rapfi"] = rapfi_manifest(cfg or Config(), ROOT)
    return result


def run_search(forest, model, cfg, noise=True, budgets=None, exploratory=None):
    forest.begin(cfg.simulations, cfg.cpuct, cfg.dirichlet_fraction if noise else 0., cfg.dirichlet_alpha,
                 cfg.fpu_reduction, cfg.root_fpu_reduction, cfg.forced_playouts if noise else 0.,
                 [] if budgets is None else list(map(int, budgets)),
                 [] if exploratory is None else list(map(bool, exploratory)))
    calls = 0
    while not forest.done():
        ids, boards, colors, last = forest.request(cfg.leaves_per_actor)
        if ids:
            probs, values = infer(model, boards, colors, last, cfg.device, cfg.amp)
            forest.complete(ids, probs, values)
        calls += 1
        if calls > max(cfg.simulations, cfg.cheap_simulations) * 3 + 10:
            raise RuntimeError("Search did not consume its simulation budget")
    return np.asarray(forest.policy(1.)), np.asarray(forest.policy(0.))


def move_temperature(ply, cfg, full_search=True):
    """Cheap searches finish games cheaply; exploration is confined to full moves."""
    if not full_search or ply >= cfg.temperature_moves:
        return 0.
    return max(cfg.temperature_floor, 2. ** (-ply / cfg.temperature_halflife))


def tempered_policy(policy, temperature):
    policy = np.asarray(policy, dtype=np.float64)
    if temperature == 0:
        result = (policy == policy.max()).astype(np.float64)
    else:
        positive = policy > 0
        result = np.zeros_like(policy)
        logs = np.log(policy[positive]) / temperature
        result[positive] = np.exp(logs - logs.max())
    return result / result.sum()


class Replay:
    def __init__(self, capacity):
        self.capacity, self.count, self.cursor = capacity, 0, 0
        self.total_added = 0
        self.board = np.zeros((capacity, 225), dtype=np.int8)
        self.color = np.zeros(capacity, dtype=np.int8)
        self.last = np.full(capacity, -1, dtype=np.int16)
        self.policy = np.zeros((capacity, 225), dtype=np.float16)
        self.value = np.zeros(capacity, dtype=np.float32)

    def append_game(self, samples, winner):
        for board, color, last, policy in samples:
            i = self.cursor
            self.board[i], self.color[i], self.last[i], self.policy[i] = board, color, last, policy
            self.value[i] = 0 if winner == 0 else (1 if color == winner else -1)
            self.cursor = (self.cursor + 1) % self.capacity
            self.count = min(self.count + 1, self.capacity)
            self.total_added += 1

    def state(self):
        return {"capacity": self.capacity, "count": self.count, "cursor": self.cursor,
                "total_added": self.total_added,
                **{name: getattr(self, name)[:self.count].copy() for name in ("board", "color", "last", "policy", "value")}}

    @classmethod
    def restore(cls, state):
        obj = cls(state["capacity"])
        obj.count, obj.cursor = state["count"], state["cursor"]
        obj.total_added = state.get("total_added", obj.count)
        for name in ("board", "color", "last", "policy", "value"):
            getattr(obj, name)[:obj.count] = state[name]
        return obj


def augment(planes, policy, codes):
    # The same D4 transform is applied to all feature planes and the policy.
    size = planes.shape[-1]
    transformed_planes = torch.empty_like(planes)
    transformed_policy = torch.empty_like(policy)
    policy_board = policy.reshape(-1, size, size)
    for code in range(8):
        mask = torch.as_tensor(codes == code, device=planes.device)
        x, p = planes[mask], policy_board[mask]
        if code >= 4:
            x, p = x.flip(-1), p.flip(-1)
        transformed_planes[mask] = x.rot90(code % 4, (-2, -1))
        transformed_policy[mask] = p.rot90(code % 4, (-2, -1)).flatten(1)
    return transformed_planes, transformed_policy


def optimize(model, optimizer, replay, rng, cfg, step, progress=None):
    ids = rng.integers(replay.count, size=cfg.batch_size)
    planes = make_planes(replay.board[ids], replay.color[ids], replay.last[ids], device=cfg.device)
    target_pi = torch.as_tensor(replay.policy[ids].astype(np.float32), device=cfg.device)
    target_pi = target_pi / target_pi.sum(-1, keepdim=True)
    target_z = torch.as_tensor(replay.value[ids], device=cfg.device)
    planes, target_pi = augment(planes, target_pi, rng.integers(8, size=cfg.batch_size))
    # Optimizer-step schedule is deterministic across exact checkpoint resume.
    # Its horizon is reduced with the lower presentations/new-position ratio.
    progress = min(1., step / cfg.schedule_steps if progress is None else progress)
    scale = cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))
    warmup = min(1., (step + 1) / max(1, cfg.warmup_steps))
    lr = cfg.learning_rate * scale * warmup
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=torch.device(cfg.device).type, dtype=torch.bfloat16,
                        enabled=cfg.amp and torch.device(cfg.device).type == "cuda"):
        logits, value, value_logits = model(planes, return_value_logits=True)
    # Normalization and both objectives stay FP32, including extreme logits.
    occupied = (planes[:, :2].sum(1) > 0).flatten(1)
    log_probs = logits.float().masked_fill(occupied, -1e9).log_softmax(-1)
    policy_loss = -(target_pi * log_probs).sum(-1).mean()
    if model.cfg.value_head == "wdl":
        labels = torch.where(target_z > 0, 0, torch.where(target_z < 0, 1, 2)).long()
        value_loss = F.cross_entropy(value_logits.float(), labels)
    else:
        value_loss = F.mse_loss(value.float(), target_z)
    loss = policy_loss + value_loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite loss")
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    result = {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
              "value_loss": float(value_loss.detach()), "value_mse": float(F.mse_loss(value.detach(), target_z)),
              "value_saturated_fraction": float((value.detach().abs() > .99).float().mean()),
              "grad_norm": float(grad), "lr": lr}
    for color, name in ((True, "black"), (False, "white")):
        mask = (planes[:, 3, 0, 0] > 0) == color
        if bool(mask.any()):
            result[name + "_value_mse"] = float(F.mse_loss(value.detach()[mask], target_z[mask]))
            result[name + "_positive_target"] = float((target_z[mask] > 0).float().mean())
    return result


def acquire(root):
    lock = root / "process.json"
    if lock.exists():
        old = json.loads(lock.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(old["pid"])
            if process.is_running() and abs(process.create_time() - old["created"]) < .1:
                raise RuntimeError(f"This run is already active: PID {old['pid']}")
        except psutil.NoSuchProcess:
            pass
        lock.unlink()
    process = psutil.Process()
    identity = {"pid": os.getpid(), "created": process.create_time(), "root": str(root), "command": process.cmdline()}
    # Exclusive creation closes the check-then-start race between supervisors.
    with lock.open("x", encoding="utf-8") as f:
        json.dump(identity, f, ensure_ascii=False)
    return identity


class Session:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root).resolve(), cfg
        cfg.validate()
        # Check before acquiring a lock or writing any budget/status files.
        if (self.root / "termination.json").exists():
            raise RuntimeError("This experiment was terminated by the user; use a new run root")
        if (self.root / "plan.json").exists():
            existing = json.loads((self.root / "plan.json").read_text(encoding="utf-8"))
            if existing.get("protocol_version") != 2 or existing["config"] != asdict(cfg) or existing["sources"] != source_manifest(cfg):
                raise RuntimeError("Frozen experiment differs; v2 requires its own run root")
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = acquire(self.root)
        self.start = time.monotonic()
        budget_path = self.root / "budget.json"
        if not budget_path.exists():
            epoch = time.time()
            atomic_json(budget_path, {"started_epoch": epoch,
                "deadline_epoch": epoch + cfg.runtime_hours * 3600,
                "runtime_hours": cfg.runtime_hours,
                "clock": "fixed wall-clock window including training, development, final evaluation and any downtime"})
        self.budget = json.loads(budget_path.read_text(encoding="utf-8"))
        if self.budget["runtime_hours"] != cfg.runtime_hours:
            raise RuntimeError("Runtime budget changed")
        self.prior_seconds = 0.
        self.wave = self.steps = self.plies = self.games = 0
        self.outcomes = {"black": 0, "white": 0, "draw": 0}
        self.last_save = self.last_dev = 0.
        self.latest_losses = {}
        self.training_positions = 0
        self.training_credit = 0.
        self.actor_step = 0
        self.actor_version = 0
        self.forks = []
        self.recent_games = deque(maxlen=2048)
        self.latest_health = {}
        self.prior_search = {"network_evaluations": 0, "simulations": 0}
        torch.set_num_threads(4)
        torch.manual_seed(cfg.seed)
        if torch.device(cfg.device).type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required by this configuration")
            torch.cuda.manual_seed_all(cfg.seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.rng = np.random.default_rng(cfg.seed)
        self.model = BoardTransformer(NetworkConfig(**cfg.network)).to(cfg.device).eval()
        self.actor = copy.deepcopy(self.model).requires_grad_(False)
        self.initial_model = copy.deepcopy(self.actor)
        self.previous_dev_model = copy.deepcopy(self.actor)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate,
                                           weight_decay=cfg.weight_decay,
                                           fused=torch.device(cfg.device).type == "cuda")
        self.forest = native().Forest(cfg.actors, cfg.seed)
        self.replay = Replay(cfg.replay_capacity)
        self.histories = [[] for _ in range(cfg.actors)]
        self.samples = [[] for _ in range(cfg.actors)]
        self.opening_lengths = [0] * cfg.actors
        self.opening_kinds = ["empty"] * cfg.actors
        self.temperature_sampled_moves = [0] * cfg.actors
        plan = {"protocol_version": 2, "config": asdict(cfg), "sources": source_manifest(cfg), "network": self.model.manifest(),
                "created_at": utc(), "random_initialization": True,
                "rules": "15x15 freestyle; >=5 wins; no pass/resign/forbidden moves/Swap2",
                "training": "Mixed search caps; full-search pruned policy targets only; terminal WDL labels; uniform replay and D4",
                "sample_accounting": "Learning credit is per newly retained full-search position, not per real move",
                "temperature": "Full search: exponential decay then greedy; cheap search: greedy without root noise",
                "openings": "Empty boards plus self-generated random prefixes and completed-selfplay forks; no evaluation bank",
                "teachers": False, "tactical_training_rewards": False,
                "tree_cache": "Actor weights frozen between publications; clear on publication/checkpoint only",
                "checkpoint_selection": "final checkpoint at fixed runtime/step cap; no test-based selection",
                "test_use": "final assessment only; paired colors; raw network and search reported separately"}
        plan_path = self.root / "plan.json"
        if plan_path.exists():
            old = json.loads(plan_path.read_text(encoding="utf-8"))
            if old["config"] != plan["config"] or old["sources"] != plan["sources"]:
                raise RuntimeError("Frozen configuration/source/asset hashes changed; refusing resume")
        else:
            atomic_json(plan_path, plan)
            torch.save({"config": cfg.network, "model": self.model.state_dict()}, self.root / "initial.pt")
        status = self.root / "status.json"
        if status.exists():
            self.prior_seconds = json.loads(status.read_text(encoding="utf-8")).get("elapsed_seconds", 0.)
        loaded = False
        for path in (self.root / "latest.pt", self.root / "previous.pt"):
            if not path.exists():
                continue
            try:
                checkpoint = torch.load(path, map_location=cfg.device, weights_only=False)
            except (OSError, RuntimeError, EOFError):
                continue
            self.restore(checkpoint)
            loaded = True
            break
        if not loaded and status.exists() and self.prior_seconds > 0:
            raise RuntimeError("Existing progress has no readable checkpoint; refusing silent restart")
        self.log({"event": "resumed" if loaded else "started", "timestamp": utc(), "pid": os.getpid(),
                  "device": cfg.device, "network": self.model.manifest(), "prior_cost_seconds": self.prior_seconds})

    def elapsed(self):
        return max(self.prior_seconds + time.monotonic() - self.start,
                   time.time() - self.budget["started_epoch"])

    def counters(self):
        return {k: self.prior_search[k] + int(v) for k, v in self.forest.counters().items()}

    def log(self, value):
        with (self.root / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False) + "\n")

    def status(self, state="running", **extra):
        value = {"state": state, "phase": "train", "updated_at": utc(), "pid": os.getpid(),
                 "identity": self.identity, "elapsed_seconds": self.elapsed(), "wave": self.wave,
                 "optimizer_steps": self.steps, "real_plies": self.plies, "completed_games": self.games,
                 "outcomes": self.outcomes, "replay_count": self.replay.count, "search": self.counters(),
                 "retained_training_positions": self.training_positions,
                 "sample_presentations": self.steps * self.cfg.batch_size,
                 "training_credit_samples": self.training_credit,
                 "actor_step": self.actor_step, "actor_version": self.actor_version,
                 "health": self.latest_health,
                 "losses": self.latest_losses, **extra}
        atomic_json(self.root / "status.json", value)
        return value

    def restore(self, checkpoint):
        if checkpoint.get("protocol_version") != 2:
            raise RuntimeError("Cannot resume a v1 training state under v2")
        self.model.load_state_dict(checkpoint["model"])
        self.actor.load_state_dict(checkpoint["actor_model"])
        self.previous_dev_model.load_state_dict(checkpoint["previous_dev_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.replay = Replay.restore(checkpoint["replay"])
        for name in ("wave", "steps", "plies", "games", "outcomes", "histories", "samples", "last_dev",
                     "training_positions", "training_credit", "actor_step", "actor_version", "forks",
                     "opening_lengths", "opening_kinds", "temperature_sampled_moves", "latest_health", "latest_losses"):
            setattr(self, name, checkpoint[name])
        self.recent_games = deque(checkpoint["recent_games"], maxlen=2048)
        self.prior_seconds = max(self.prior_seconds, checkpoint["elapsed_seconds"])
        self.last_save = checkpoint["elapsed_seconds"]
        self.prior_search = checkpoint["search"]
        self.rng.bit_generator.state = checkpoint["numpy_rng"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
        for i, history in enumerate(self.histories):
            self.forest.set_history(i, history)
        self.forest.set_rng_state(checkpoint["search_rng"])

    def save(self, final=False):
        self.forest.clear_trees()
        checkpoint = {"protocol_version": 2, "model": self.model.state_dict(),
                      "actor_model": self.actor.state_dict(), "previous_dev_model": self.previous_dev_model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "network": self.cfg.network, "config": asdict(self.cfg), "replay": self.replay.state(),
                      **{name: getattr(self, name) for name in ("wave", "steps", "plies", "games", "outcomes", "histories", "samples", "last_dev",
                          "training_positions", "training_credit", "actor_step", "actor_version", "forks",
                          "opening_lengths", "opening_kinds", "temperature_sampled_moves", "latest_health", "latest_losses")},
                      "recent_games": list(self.recent_games),
                      "elapsed_seconds": self.elapsed(), "search": self.counters(),
                      "numpy_rng": self.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if torch.device(self.cfg.device).type == "cuda" else None,
                      "search_rng": self.forest.rng_state()}
        temp = self.root / "latest.pt.tmp"
        torch.save(checkpoint, temp)
        if (self.root / "latest.pt").exists():
            shutil.copyfile(self.root / "latest.pt", self.root / "previous.pt.tmp")
            os.replace(self.root / "previous.pt.tmp", self.root / "previous.pt")
        os.replace(temp, self.root / "latest.pt")
        # Small, observational model snapshots. They never select the final weight.
        snapshot = self.root / "snapshots"
        snapshot.mkdir(exist_ok=True)
        if final or self.elapsed() // 43200 > self.last_save // 43200:
            torch.save({"config": self.cfg.network, "model": self.model.state_dict(),
                        "real_plies": self.plies, "optimizer_steps": self.steps},
                       snapshot / ("final.pt" if final else f"hour_{int(self.elapsed() // 3600):03d}.pt"))
        self.last_save = self.elapsed()
        self.status(checkpoint=str(self.root / "latest.pt"))

    def reset_game(self, slot):
        """State diversity uses only our own trajectories and independent RNG."""
        draw = self.rng.random()
        history, kind = [], "empty"
        if draw < self.cfg.fork_probability and self.forks:
            history = list(self.forks[int(self.rng.integers(len(self.forks)))])
            kind = "selfplay_fork"
        elif self.cfg.fork_probability <= draw < self.cfg.fork_probability + self.cfg.random_opening_probability:
            length = int(self.rng.integers(1, self.cfg.random_opening_max_moves + 1)) if self.cfg.random_opening_max_moves else 0
            history = self.rng.choice(225, size=length, replace=False).tolist()
            kind = "random_prefix" if history else "empty"
            # These are newly generated legal opening moves, counted as cost.
            self.plies += len(history)
        self.histories[slot], self.samples[slot] = history, []
        self.opening_lengths[slot], self.opening_kinds[slot] = len(history), kind
        self.temperature_sampled_moves[slot] = 0
        self.forest.set_history(slot, history)

    def health(self):
        result = {"warnings": [], "recent_games": len(self.recent_games)}
        if self.recent_games:
            winners = np.array([x[0] for x in self.recent_games])
            lengths = np.array([x[1] for x in self.recent_games])
            total_lengths = np.array([x[2] for x in self.recent_games])
            sampled = np.array([x[3] for x in self.recent_games])
            result.update(recent_black_win_fraction=float((winners == 1).mean()),
                          recent_mean_played_plies=float(lengths.mean()),
                          recent_games_within_temperature_cutoff=float((total_lengths <= self.cfg.temperature_moves).mean()),
                          recent_temperature_sampled_move_fraction=float(sampled.sum() / lengths.sum()))
            if len(winners) >= 256 and ((winners == 1).mean() > .98 or (winners == -1).mean() > .98):
                result["warnings"].append("recent_selfplay_color_imbalance")
        if self.replay.count:
            # Deterministic evenly-spaced diagnostics must not consume training RNG.
            ids = np.linspace(0, self.replay.count - 1, min(4096, self.replay.count), dtype=int)
            for color, name in ((1, "black"), (-1, "white")):
                selected = ids[self.replay.color[ids] == color]
                if not len(selected):
                    continue
                p = self.replay.policy[selected].astype(np.float64)
                p /= p.sum(-1, keepdims=True)
                ent = -(p * np.log(np.maximum(p, 1e-30))).sum(-1)
                result[name] = {"n": len(selected), "target_policy_entropy": float(ent.mean()),
                                "target_top1": float(p.max(-1).mean()),
                                "positive_value_target": float((self.replay.value[selected] > 0).mean()),
                                "color_only_mse": float(np.square(self.replay.value[selected] - color).mean())}
                if ent.mean() > math.log(80) and ((self.replay.value[selected] > 0).mean() < .02):
                    result["warnings"].append(name + "_diffuse_losing_policy_targets")
        self.latest_health = result
        return result

    def run_wave(self):
        boards, colors, last, ply, _, _ = self.forest.states()
        full = self.rng.random(self.cfg.actors) < self.cfg.full_search_probability
        budgets = np.where(full, self.cfg.simulations, min(self.cfg.cheap_simulations, self.cfg.simulations))
        policy, greedy = run_search(self.forest, self.actor, self.cfg, budgets=budgets, exploratory=full)
        targets = np.asarray(self.forest.policy(1., self.cfg.policy_target_pruning))
        actions = []
        for i in range(self.cfg.actors):
            temperature = move_temperature(ply[i], self.cfg, full[i])
            self.temperature_sampled_moves[i] += int(temperature > 0)
            p = tempered_policy(policy[i], temperature) if temperature > 0 else greedy[i]
            action = int(self.rng.choice(225, p=p.astype(np.float64) / p.sum(dtype=np.float64)))
            if full[i]:
                self.samples[i].append((boards[i].copy(), colors[i], last[i], targets[i].astype(np.float16)))
            self.histories[i].append(action)
            actions.append(action)
        self.forest.advance(actions)
        _, _, _, counts, winners, done = self.forest.states()
        for i, ended in enumerate(done):
            if not ended:
                continue
            self.replay.append_game(self.samples[i], winners[i])
            self.outcomes["black" if winners[i] == 1 else "white" if winners[i] == -1 else "draw"] += 1
            self.games += 1
            with (self.root / "selfplay.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"game": self.games, "wave": self.wave + 1, "history": self.histories[i],
                                    "winner": winners[i], "plies": counts[i],
                                    "opening_plies": self.opening_lengths[i], "opening_kind": self.opening_kinds[i],
                                    "played_plies": counts[i] - self.opening_lengths[i],
                                    "temperature_sampled_moves": self.temperature_sampled_moves[i],
                                    "retained_positions": len(self.samples[i]), "actor_version_at_finish": self.actor_version,
                                    "terminal_reason": "five_in_row" if winners[i] else "full_board_draw"}) + "\n")
            self.recent_games.append((winners[i], counts[i] - self.opening_lengths[i],
                                      counts[i], self.temperature_sampled_moves[i]))
            if len(self.histories[i]) > 4:
                # Strict prefix of a verified nonterminal trajectory; never use test positions.
                length = int(self.rng.integers(2, len(self.histories[i])))
                prefix = self.histories[i][:length]
                if len(self.forks) < self.cfg.fork_capacity:
                    self.forks.append(prefix)
                else:
                    self.forks[int(self.rng.integers(len(self.forks)))] = prefix
            self.reset_game(i)
        self.plies += self.cfg.actors
        self.wave += 1
        added = self.replay.total_added - self.training_positions
        self.training_positions = self.replay.total_added
        self.training_credit = min(self.cfg.train_credit_max_samples,
                                   self.training_credit + added * self.cfg.train_sample_ratio)
        if self.replay.count >= self.cfg.min_replay:
            updates = min(self.cfg.steps_per_wave, int(self.training_credit // self.cfg.batch_size),
                          self.cfg.max_train_steps - self.steps)
            for _ in range(updates):
                self.latest_losses = optimize(self.model, self.optimizer, self.replay, self.rng, self.cfg, self.steps)
                self.steps += 1
                self.training_credit -= self.cfg.batch_size
        if self.steps - self.actor_step >= self.cfg.actor_publish_steps:
            self.actor.load_state_dict(self.model.state_dict())
            self.actor_step, self.actor_version = self.steps, self.actor_version + 1
            self.forest.clear_trees()
        if self.wave % 100 == 0:
            self.health()

    def run(self, max_waves=None):
        stop = self.wave + max_waves if max_waves is not None else math.inf
        deadline = (self.cfg.runtime_hours - self.cfg.final_eval_reserve_hours) * 3600
        self.status()
        try:
            while self.wave < stop and self.elapsed() < deadline and self.steps < self.cfg.max_train_steps:
                self.run_wave()
                status = self.status()
                if self.wave % 10 == 0:
                    self.log({"event": "progress", **status})
                    print(json.dumps({k: status[k] for k in ("wave", "real_plies", "completed_games", "optimizer_steps", "elapsed_seconds", "losses")}), flush=True)
                if self.elapsed() - self.last_save >= self.cfg.checkpoint_seconds:
                    self.save()
                if self.elapsed() - self.last_dev >= self.cfg.dev_eval_seconds and self.steps > 0:
                    self.save()
                    from .evaluate import dev_evaluate
                    result = dev_evaluate(self.model, self.cfg, initial=self.initial_model,
                                          previous=self.previous_dev_model)
                    self.previous_dev_model.load_state_dict(self.model.state_dict())
                    self.health()
                    self.last_dev = self.elapsed()
                    self.log({"event": "development", "optimizer_steps": self.steps, "elapsed_seconds": self.elapsed(), **result})
            # Partial game buffers are saved, never mislabeled as draws.
            capped = self.elapsed() >= deadline or self.steps >= self.cfg.max_train_steps
            self.save(final=capped)
            self.status("trained" if capped else "bounded_check_complete",
                        stop_reason="runtime_cap" if self.elapsed() >= deadline else "step_cap" if capped else "wave_cap")
        except BaseException as error:
            self.log({"event": "failure", "timestamp": utc(), "error": repr(error), "traceback": traceback.format_exc()})
            self.status("failed", error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-waves", type=int)
    args = parser.parse_args()
    cfg = Config(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    Session(args.root, cfg).run(args.max_waves)


if __name__ == "__main__":
    main()
