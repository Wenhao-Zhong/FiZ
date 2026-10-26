"""Separated development diagnostics and fixed, deadline-bounded final evaluation.

Every game has its own RNG and search forest. Batching never determines which
game's random stream is consumed. Final tests are not opened by development.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from .model import BoardTransformer, NetworkConfig, infer
from .search import native
from .train import ROOT


class Deadline(Exception):
    pass


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise Deadline("evaluation wall-clock deadline reached")


def _rng_list(rng, n):
    if isinstance(rng, (list, tuple)):
        if len(rng) != n:
            raise ValueError("One RNG is required per game")
        return rng
    return [rng] * n  # backward-compatible single-game/probe caller


def _moves_and_diagnostics(model, histories, cfg, rng, search=True, deadline=float("inf")):
    _check_deadline(deadline)
    if not histories:
        return [], []
    rngs = _rng_list(rng, len(histories))
    seeds = [int(r.integers(0, 2**63 - 1)) for r in rngs]
    forests = [native().Forest(1, seed) for seed in seeds]
    for forest, history in zip(forests, histories):
        forest.set_history(0, history)
        if forest.states()[-1][0]:
            raise ValueError("Cannot evaluate a terminated history")
    raw_policies, raw_values = [None] * len(forests), [None] * len(forests)
    if search:
        for forest in forests:
            forest.begin(cfg.eval_simulations, cfg.cpuct, 0., cfg.dirichlet_alpha,
                         getattr(cfg, "fpu_reduction", .2), getattr(cfg, "root_fpu_reduction", .2), 0.)
        while not all(f.done() for f in forests):
            _check_deadline(deadline)
            requests, boards, colors, last = [], [], [], []
            for i, forest in enumerate(forests):
                if forest.done():
                    continue
                ids, b, c, l = forest.request(cfg.leaves_per_actor)
                if ids:
                    requests.append((i, ids, len(boards), len(ids)))
                    boards.extend(b); colors.extend(c); last.extend(l)
            if not boards:
                continue
            probabilities, values = infer(model, np.stack(boards), colors, last, cfg.device, cfg.amp)
            _check_deadline(deadline)
            for i, ids, start, count in requests:
                if raw_policies[i] is None:
                    raw_policies[i], raw_values[i] = probabilities[start], float(values[start])
                forests[i].complete(ids, probabilities[start:start+count], values[start:start+count])
        choices = [np.asarray(f.policy(0.))[0] for f in forests]
    else:
        states = [f.states() for f in forests]
        probabilities, values = infer(model, np.stack([s[0][0] for s in states]),
                                      [s[1][0] for s in states], [s[2][0] for s in states], cfg.device, cfg.amp)
        _check_deadline(deadline)
        raw_policies, raw_values = probabilities, values
        choices = [(p == p.max()).astype(np.float64) for p in probabilities]
    actions, diagnostics = [], []
    for history, p, v, choice, rng_i in zip(histories, raw_policies, raw_values, choices, rngs):
        choice = np.asarray(choice, dtype=np.float64)
        actions.append(int(rng_i.choice(225, p=choice / choice.sum())))
        positive = p[p > 0]
        diagnostics.append({"ply": len(history), "color": 1 if len(history) % 2 == 0 else -1,
            "raw_policy_entropy": float(-(positive * np.log(positive)).sum()),
            "raw_policy_top1": float(p.max()), "raw_value": float(v)})
    return actions, diagnostics


def moves(model, histories, cfg, rng, search=True, deadline=float("inf")):
    return _moves_and_diagnostics(model, histories, cfg, rng, search, deadline)[0]


def opponent_moves(kind, forest, indices, rngs, initial, cfg, search, rapfi=None, deadline=float("inf")):
    boards, colors, _, _, _, _ = forest.states()
    if kind == "initial":
        raise RuntimeError("Initial actions require complete histories")
    if kind == "one_ply":
        from .tactics import choose_tactical
    result = []
    for i in indices:
        _check_deadline(deadline)
        if kind == "random":
            action = rngs[i].choice(np.flatnonzero(boards[i] == 0))
        elif kind == "one_ply":
            state = SimpleNamespace(board=boards[i], to_play=0 if colors[i] == 1 else 1, legal_mask=boards[i] == 0)
            action = choose_tactical(state, rngs[i])
        elif kind == "rapfi":
            if rapfi is None:
                raise RuntimeError("Rapfi adapter unavailable")
            action = rapfi.move(boards[i], colors[i])
        else:
            raise ValueError(kind)
        _check_deadline(deadline)
        result.append(int(action))
    return result


def _complete_pairs(games):
    pairs = {}
    for game in games:
        pair, color, score = game["pair"], game["color"], game["score"]
        if color not in (1, -1) or score not in (0, .5, 1):
            raise ValueError("Invalid paired game record")
        group = pairs.setdefault(pair, {})
        if color in group:
            raise ValueError("Duplicate pair/color game")
        group[color] = float(score)
    return {k: v for k, v in pairs.items() if set(v) == {1, -1}}


def _bounded_interval(values, low, high, seed):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    radius = (high - low) * math.sqrt(math.log(40.) / (2 * len(values)))
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(-1)
    return {"mean": mean, "ci95": [max(low, mean-radius), min(high, mean+radius)],
        "ci_method": "Hoeffding 95% bounded-mean interval; independent opening pairs are the units",
        "bootstrap_ci95": np.quantile(boot, [.025, .975]).tolist(),
        "bootstrap_note": "descriptive only; percentile bootstrap can collapse at all-win/all-loss boundaries",
        "uncertainty_scope": "evaluation opening pairs, not variation across training seeds"}


def paired_interval(games, seed=2718):
    pairs = _complete_pairs(games)
    if not pairs:
        return None
    stats = _bounded_interval([np.mean(list(p.values())) for p in pairs.values()], 0., 1., seed)
    stats.update(completed_pairs=len(pairs), unpaired_games_excluded=len(games)-2*len(pairs),
                 black=float(np.mean([p[1] for p in pairs.values()])), white=float(np.mean([p[-1] for p in pairs.values()])))
    return stats


def _arena_result(games, requested, error=None):
    return dict(summary=paired_interval(games), completed_games=len(games), requested_games=requested,
                incomplete=len(games) != requested or error is not None, error=error, games=games)


def arena(model, cfg, openings, kind, initial=None, search=True, deadline=float("inf"),
          raw_path=None, batch=16, rapfi=None, collect_diagnostics=False):
    from .rules import reference_replay
    jobs = [(i, color, list(opening)) for i, opening in enumerate(openings) for color in (1, -1)]
    if not jobs:
        return _arena_result([], 0)
    if kind == "initial" and initial is None:
        return _arena_result([], len(jobs), "comparison model unavailable")
    for _, _, history in jobs:
        if reference_replay(history).terminated:
            raise ValueError("Evaluation opening already terminated")
    games = []
    if raw_path and Path(raw_path).exists():
        games = [json.loads(line) for line in Path(raw_path).read_text("utf-8").splitlines() if line.strip()]
        _complete_pairs(games)
        for g in games:
            if not 0 <= g["pair"] < len(openings) or g["history"][:len(openings[g["pair"]])] != list(openings[g["pair"]]):
                raise ValueError("Saved games do not match this opening bank")
            ref = reference_replay(g["history"])
            winner = 0 if ref.winner == -1 else 1 if ref.winner == 0 else -1
            score = .5 if winner == 0 else float(winner == g["color"])
            if not ref.terminated or winner != g["winner"] or score != g["score"]:
                raise ValueError("Invalid saved complete game")
    completed = {(g["pair"], g["color"]) for g in games}
    pending_jobs = [j for j in jobs if j[:2] not in completed]
    if not pending_jobs:
        return _arena_result(games, len(jobs))
    if time.monotonic() >= deadline:
        return _arena_result(games, len(jobs), "deadline")
    size = min(batch, len(pending_jobs))
    forest = native().Forest(size, 4242)
    histories, metadata = [[] for _ in range(size)], [None] * size
    agent_rngs, opponent_rngs, traces = [None] * size, [None] * size, [[] for _ in range(size)]
    cursor = 0

    def assign(slot):
        nonlocal cursor
        if cursor >= len(pending_jobs):
            metadata[slot] = None
            return
        pair, color, history = pending_jobs[cursor]; cursor += 1
        histories[slot], metadata[slot], traces[slot] = list(history), (pair, color, len(history)), []
        seed = int(getattr(cfg, "eval_seed", 271828)) + pair * 16 + int(color == -1) * 4
        agent_rngs[slot], opponent_rngs[slot] = np.random.default_rng(seed), np.random.default_rng(seed + 1)
        forest.set_history(slot, histories[slot])

    for i in range(size):
        assign(i)
    try:
        while any(x is not None for x in metadata):
            _check_deadline(deadline)
            _, colors, *_ = forest.states()
            agent, opponents = [], []
            for i, data in enumerate(metadata):
                if data is not None:
                    (agent if colors[i] == data[1] else opponents).append(i)
            actions = [-1] * size
            if agent:
                chosen, diagnostics = _moves_and_diagnostics(model, [histories[i] for i in agent], cfg,
                    [agent_rngs[i] for i in agent], search, deadline)
                for i, action, diagnostic in zip(agent, chosen, diagnostics):
                    actions[i] = action
                    if collect_diagnostics:
                        traces[i].append(diagnostic)
            if opponents:
                if kind == "initial":
                    chosen = moves(initial, [histories[i] for i in opponents], cfg,
                                   [opponent_rngs[i] for i in opponents], search, deadline)
                else:
                    chosen = opponent_moves(kind, forest, opponents, opponent_rngs, initial, cfg, search, rapfi, deadline)
                for i, action in zip(opponents, chosen):
                    actions[i] = action
            _check_deadline(deadline)
            forest.advance(actions)
            _, _, _, counts, winners, ended = forest.states()
            for i, data in enumerate(metadata):
                if data is None:
                    continue
                histories[i].append(actions[i])
                if ended[i]:
                    ref = reference_replay(histories[i])
                    winner = 0 if ref.winner == -1 else 1 if ref.winner == 0 else -1
                    if not ref.terminated or winner != winners[i]:
                        raise AssertionError("Native and independent arena outcomes disagree")
                    pair, color, opening_len = data
                    score = .5 if winner == 0 else float(winner == color)
                    record = {"pair": pair, "color": color, "score": score, "winner": winner,
                        "history": histories[i], "plies": counts[i], "opening_plies": opening_len,
                        "terminal_reason": "five_in_row" if winner else "full_board_draw"}
                    if collect_diagnostics:
                        record["network_diagnostics"] = [dict(d, target_value=2*score-1) for d in traces[i]]
                    games.append(record)
                    if raw_path:
                        with Path(raw_path).open("a", encoding="utf-8") as f:
                            f.write(json.dumps(record) + "\n")
                    assign(i)
    except Deadline:
        return _arena_result(games, len(jobs), "deadline")
    except Exception as error:
        if kind != "rapfi":
            raise
        return _arena_result(games, len(jobs), f"Rapfi error: {type(error).__name__}: {error}")
    return _arena_result(games, len(jobs))


def _diagnostic_summary(records, target_source):
    output = {"target_source": target_source, "by_color": {}}
    for color, name in ((1, "black"), (-1, "white")):
        rows = [x for x in records if x["color"] == color]
        if not rows:
            output["by_color"][name] = {"n": 0}
            continue
        values, target = np.asarray([x["raw_value"] for x in rows]), np.asarray([x["target_value"] for x in rows])
        probability, label = (values + 1) / 2, (target + 1) / 2
        bins, ece = [], 0.
        for i in range(10):
            mask = (probability >= i/10) & (probability < (i+1)/10 if i < 9 else probability <= 1)
            if mask.any():
                predicted, observed = float(probability[mask].mean()), float(label[mask].mean())
                ece += float(mask.mean()) * abs(predicted-observed)
                bins.append({"lower": i/10, "upper": (i+1)/10, "n": int(mask.sum()), "mean_prediction": predicted, "observed_target": observed})
        output["by_color"][name] = {"n": len(rows), "value_mse": float(np.mean((values-target)**2)),
            "brier_score": float(np.mean((probability-label)**2)), "ece_10_bins": ece,
            "value_mean": float(values.mean()), "target_mean": float(target.mean()),
            "saturated_value_fraction": float(np.mean(np.abs(values) >= .98)),
            "confident_wrong_fraction": float(np.mean(values * target <= -.9)),
            "raw_policy_entropy_mean": float(np.mean([x["raw_policy_entropy"] for x in rows])),
            "raw_policy_top1_mean": float(np.mean([x["raw_policy_top1"] for x in rows])), "calibration_bins": bins}
    return output


def _tactical_result(items, records):
    order = {item["id"]: i for i, item in enumerate(items)}
    records = sorted(records, key=lambda row: order[row["id"]])
    summary = {kind: {} for kind in ("immediate_win", "forced_block")}
    for kind in summary:
        for color, name in ((0, "black"), (1, "white")):
            chosen = [x["correct"] for x in records if x["kind"] == kind and x["color"] == color]
            summary[kind][name] = {"accuracy": float(np.mean(chosen)) if chosen else None, "n": len(chosen)}
    return {"completed": len(records), "requested": len(items), "incomplete": len(records) != len(items),
            "summary": summary, "records": records}


def evaluate_tactics(model, cfg, items, search, deadline, seed=98123, prior=None, on_progress=None):
    from .rules import reference_replay
    by_id = {item["id"]: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("Duplicate tactical item IDs")
    records = list(prior.get("records", [])) if prior else []
    if prior and (prior.get("requested") != len(items) or prior.get("completed") != len(records)):
        raise ValueError("Saved tactical counts do not match the suite")
    completed_ids = set()
    for record in records:
        item = by_id.get(record.get("id"))
        if item is None or item["id"] in completed_ids:
            raise ValueError("Saved tactical record has unknown or duplicate ID")
        if record.get("kind") != item["kind"] or record.get("color") != item["color"]:
            raise ValueError("Saved tactical record belongs to a different stratum")
        action = record.get("action")
        state = reference_replay(item["history"])
        if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < 225 or not state.legal_mask[action]:
            raise ValueError("Saved tactical action is illegal")
        if record.get("correct") != (action in item["correct_actions"]):
            raise ValueError("Saved tactical score disagrees with the suite")
        completed_ids.add(item["id"])
    pending = [(index, item) for index, item in enumerate(items) if item["id"] not in completed_ids]
    for start in range(0, len(pending), 16):
        chunk = pending[start:start+16]
        batch = [item for _, item in chunk]
        # Preserve each original item seed even when earlier items are resumed.
        rngs = [np.random.default_rng(seed+index) for index, _ in chunk]
        try:
            actions = moves(model, [item["history"] for item in batch], cfg, rngs, search, deadline)
        except Deadline:
            break
        for item, action in zip(batch, actions):
            records.append({"id": item["id"], "kind": item["kind"], "color": item["color"], "action": action, "correct": action in item["correct_actions"]})
        if on_progress is not None:
            on_progress(_tactical_result(items, records))
    return _tactical_result(items, records)


def tactics(model, cfg, search, deadline, prior=None, on_progress=None):
    from .tactics import load_suite
    suite = load_suite(ROOT / "data/gomoku_tactics.json")
    if prior and prior.get("suite_sha256") != suite["sha256"]:
        raise ValueError("Saved tactical results belong to a different suite")
    def progress(value):
        if on_progress is not None:
            on_progress(dict(value, suite_sha256=suite["sha256"]))
    return dict(evaluate_tactics(model, cfg, suite["entries"], search, deadline,
        prior=prior, on_progress=progress), suite_sha256=suite["sha256"])


def dev_evaluate(model, cfg, initial=None, previous=None):
    from .dev_suite import generate_dev_suite
    bank = json.loads((ROOT / "data/gomoku_openings.json").read_text(encoding="utf-8"))
    openings = [x["actions"] for x in bank["openings"] if x["split"] == "dev"][:cfg.dev_pairs]
    suite = generate_dev_suite(getattr(cfg, "dev_tactics_per_stratum", 16), getattr(cfg, "dev_tactics_seed", 910241))
    deadline = time.monotonic() + getattr(cfg, "dev_eval_max_seconds", 600)
    result = {"development_suite_sha256": suite["sha256"], "tactics": {}, "comparisons": {},
              "purpose": "diagnostics only; never checkpoint selection or optimizer labels"}
    for search in (False, True):
        mode = "mcts" if search else "raw"
        result["tactics"][mode] = evaluate_tactics(model, cfg, suite["entries"], search, deadline)
    certified = []
    for start in range(0, len(suite["value_cases"]), 16):
        batch = suite["value_cases"][start:start+16]
        try:
            _, diagnostics = _moves_and_diagnostics(model, [x["history"] for x in batch], cfg,
                [np.random.default_rng(88121+start+i) for i in range(len(batch))], False, deadline)
        except Deadline:
            break
        certified.extend(dict(d, target_value=item["certified_value"]) for item, d in zip(batch, diagnostics))
    result["certified_value_diagnostics"] = dict(_diagnostic_summary(certified, suite["value_target_source"]),
        completed=len(certified), requested=len(suite["value_cases"]), incomplete=len(certified) != len(suite["value_cases"]))
    for search in (False, True):
        mode = "mcts" if search else "raw"
        anchor = arena(model, cfg, openings, "one_ply", search=search, deadline=deadline, collect_diagnostics=True)
        diagnostics = [d for game in anchor["games"] for d in game.get("network_diagnostics", [])]
        result[f"{mode}_game_value_diagnostics"] = _diagnostic_summary(diagnostics,
            "actual terminal outcomes of completed development games against the fixed one-ply opponent; conditional on that opponent and this policy")
        anchor.pop("games", None)
        result[mode] = anchor
        for name, opponent in (("initial", initial), ("previous", previous)):
            if opponent is None:
                continue
            comparison = arena(model, cfg, openings[:getattr(cfg, "dev_history_pairs", 8)], "initial", initial=opponent, search=search, deadline=deadline)
            comparison.pop("games", None)
            result["comparisons"][f"{mode}_vs_{name}"] = comparison
    result["incomplete"] = (any(x["incomplete"] for x in result["tactics"].values())
        or result["certified_value_diagnostics"]["incomplete"] or any(result[m]["incomplete"] for m in ("raw", "mcts"))
        or any(x["incomplete"] for x in result["comparisons"].values()))
    result["basic_tactics_gate_passed"] = _tactics_gate(result["tactics"]["raw"], getattr(cfg, "basic_tactics_threshold", .9))
    return result


def improvement(final, baseline, seed=313):
    a, b = _complete_pairs(final), _complete_pairs(baseline)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    values = [np.mean(list(a[k].values())) - np.mean(list(b[k].values())) for k in common]
    return dict(_bounded_interval(values, -1., 1., seed), paired_openings=len(common))


def _tactics_gate(result, threshold):
    return bool(result["requested"] > 0 and result["completed"] == result["requested"] and all(
        group["n"] > 0 and group["accuracy"] is not None and group["accuracy"] >= threshold
        for kind in result["summary"].values() for group in kind.values()))


def _rapfi_adapter(executable, config, deadline):
    from .rapfi import Rapfi
    class BoundedRapfi(Rapfi):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except BaseException:
                # Rapfi.__init__ starts its child before the protocol handshake.
                # A deadline/handshake error must not leak that subprocess.
                if hasattr(self, "proc") and self.proc.poll() is None:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                raise

        def wait(self, pattern, timeout):
            _check_deadline(deadline)
            return super().wait(pattern, min(timeout, max(.001, deadline-time.monotonic())))
    _check_deadline(deadline)
    return BoundedRapfi(executable, config, milliseconds=100)


def final_evaluate(root, cfg, deadline):
    from dataclasses import asdict
    from .train import atomic_json, utc
    from .assets import rapfi_manifest, rapfi_paths
    root = Path(root)
    checkpoint_path = root / "latest.pt"
    initial_path = root / "initial.pt"
    identity = {"checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "initial_checkpoint_sha256": hashlib.sha256(initial_path.read_bytes()).hexdigest(),
                "config": asdict(cfg), "evaluation_version": 2,
                "rapfi_assets": rapfi_manifest(cfg, ROOT)}
    partial = root / "assessment.partial.json"
    prior = {}
    if partial.exists():
        prior = json.loads(partial.read_text("utf-8"))
        if prior.get("identity") != identity:
            raise RuntimeError("Existing evaluation belongs to a different checkpoint/configuration")
    elif any(root.glob("raw_*.jsonl")) or any(root.glob("mcts_*.jsonl")):
        raise RuntimeError("Evaluation JSONL exists without its identity manifest; refusing unbound reuse")
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device, weights_only=False)
    initial_ckpt = torch.load(initial_path, map_location=cfg.device, weights_only=False)
    model = BoardTransformer(NetworkConfig(**cfg.network)).to(cfg.device).eval()
    initial = BoardTransformer(NetworkConfig(**cfg.network)).to(cfg.device).eval()
    model.load_state_dict(checkpoint["model"]); initial.load_state_dict(initial_ckpt["model"])
    bank = json.loads((ROOT / "data/gomoku_openings.json").read_text(encoding="utf-8"))
    openings = [x["actions"] for x in bank["openings"] if x["split"] == "test"][:cfg.final_pairs]
    result = {"started_at": prior.get("started_at", utc()), "fixed_checkpoint": "latest.pt", "identity": identity,
              "comparisons": {}, "tactics": dict(prior.get("tactics", {})),
              "professional_strength_established": False, "single_seed_h1_confirmed": False}
    if prior:
        result["resumed_at"] = [*prior.get("resumed_at", []), utc()]
    atomic_json(partial, result)
    executable, engine_config = rapfi_paths(cfg, ROOT)
    result["rapfi_anchor"] = {"time_per_move_limit_ms": 100, "actual_time_may_be_shorter": True,
        "executable": str(executable), "config": str(engine_config), "tournament_equivalence": False}
    for search in (False, True):
        mode = "mcts" if search else "raw"
        for name, network in (("initial", initial), ("final", model)):
            tag = f"{mode}_{name}"
            def save_tactics(value):
                result["tactics"][tag] = value
                atomic_json(partial, result)
            result["tactics"][tag] = tactics(network, cfg, search, deadline, result["tactics"].get(tag), save_tactics)
            atomic_json(partial, result)
        for kind, network, label in (("initial", model, "final_vs_initial"), ("one_ply", model, "final_vs_one_ply"),
                                     ("one_ply", initial, "initial_vs_one_ply"), ("random", model, "final_vs_random"), ("random", initial, "initial_vs_random")):
            tag = f"{mode}_{label}"
            result["comparisons"][tag] = arena(network, cfg, openings, kind, initial=initial, search=search, deadline=deadline, raw_path=root / f"{tag}.jsonl")
            atomic_json(partial, result)
        for name, network in (("initial", initial), ("final", model)):
            tag = f"{mode}_{name}_vs_rapfi_100ms"
            rapfi = None
            comparison = arena(network, cfg, openings[:cfg.rapfi_final_pairs], "rapfi", search=search,
                deadline=-float("inf"), raw_path=root / f"{tag}.jsonl")
            try:
                if comparison["incomplete"]:
                    _check_deadline(deadline)
                    if executable is None or engine_config is None or not executable.is_file() or not engine_config.is_file():
                        raise FileNotFoundError("Rapfi executable/config unavailable")
                    rapfi = _rapfi_adapter(executable, engine_config, deadline)
                    comparison = arena(network, cfg, openings[:cfg.rapfi_final_pairs], "rapfi", search=search,
                        deadline=deadline, raw_path=root / f"{tag}.jsonl", rapfi=rapfi)
            except Exception as error:
                comparison = _arena_result(comparison["games"], 2*min(len(openings), cfg.rapfi_final_pairs),
                                           f"Rapfi unavailable: {type(error).__name__}: {error}")
            finally:
                if rapfi is not None:
                    try:
                        with (root / f"{tag}.engine.log").open("a", encoding="utf-8") as stream:
                            stream.write(f"\n# Evaluation attempt {utc()} ({tag})\n")
                            stream.write("\n".join(rapfi.transcript) + "\n")
                    finally:
                        rapfi.close()
            result["comparisons"][tag] = comparison
            atomic_json(partial, result)
        for name, network in (("initial", initial), ("final", model)):
            tag = f"{mode}_empty_{name}_vs_one_ply"
            result["comparisons"][tag] = arena(network, cfg, [[]]*cfg.empty_pairs, "one_ply", search=search, deadline=deadline, raw_path=root / f"{tag}.jsonl")
        for opponent in ("one_ply", "random", "rapfi_100ms"):
            result[f"{mode}_{opponent}_improvement"] = improvement(result["comparisons"][f"{mode}_final_vs_{opponent}"]["games"], result["comparisons"][f"{mode}_initial_vs_{opponent}"]["games"])
        atomic_json(partial, result)
    result["basic_tactics_gate_passed"] = _tactics_gate(result["tactics"]["raw_final"], getattr(cfg, "basic_tactics_threshold", .9))
    result["mcts_basic_tactics_gate_passed"] = _tactics_gate(result["tactics"]["mcts_final"], getattr(cfg, "basic_tactics_threshold", .9))
    result["incomplete"] = any(c["incomplete"] for c in result["comparisons"].values()) or any(t["incomplete"] for t in result["tactics"].values())
    for mode in ("raw", "mcts"):
        initial_score = result["comparisons"][f"{mode}_final_vs_initial"]["summary"]
        delta = result[f"{mode}_one_ply_improvement"]
        result[f"{mode}_single_seed_learning_evidence"] = bool(not result["incomplete"] and initial_score and initial_score["ci95"][0] > .55 and delta and delta["ci95"][0] > 0)
    result["single_seed_learning_evidence"] = result["raw_single_seed_learning_evidence"] or result["mcts_single_seed_learning_evidence"]
    result["completed_at"] = utc()
    result["cost"] = {"training_seconds": checkpoint["elapsed_seconds"], "real_plies": checkpoint["plies"], "completed_games": checkpoint["games"], "optimizer_steps": checkpoint["steps"], "search": checkpoint["search"]}
    atomic_json(root / "assessment.json", result)
    return result
