"""Supervise the fixed training window then final assessment within its runtime cap."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from az_gomoku.train import Config, atomic_json, utc, source_manifest


def freeze_config(root, cfg):
    """Run-local immutable configuration, independent of later preset edits."""
    path = root / "config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != asdict(cfg):
            raise RuntimeError("Run-local frozen configuration mismatch")
    else:
        atomic_json(path, asdict(cfg))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if (root / "termination.json").exists():
        raise RuntimeError("This experiment was terminated by the user; use a new run root")
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    cfg = Config(**json.loads(config_path.read_text(encoding="utf-8")))
    cfg.validate()
    plan_path = root / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("protocol_version") != 2 or plan["config"] != asdict(cfg) or plan["sources"] != source_manifest(cfg):
            raise RuntimeError("Frozen source/config/asset mismatch")
    prior = root / "supervisor.json"
    if prior.exists():
        import psutil
        value = json.loads(prior.read_text(encoding="utf-8"))
        try:
            process = psutil.Process(value["supervisor_pid"])
            if process.is_running() and abs(process.create_time() - value["created"]) < .1:
                raise RuntimeError("A matching supervisor is already active")
        except psutil.NoSuchProcess:
            pass
        # A crashed supervisor can leave a live trainer. Never start a second.
        if value.get("phase") == "train":
            child_identity = root / "process.json"
            if child_identity.exists():
                identity = json.loads(child_identity.read_text(encoding="utf-8"))
                try:
                    child = psutil.Process(identity["pid"])
                    if child.is_running() and abs(child.create_time() - identity["created"]) < .1:
                        raise RuntimeError("The old training child is still active")
                except psutil.NoSuchProcess:
                    pass
    config_path = freeze_config(root, cfg)
    import psutil
    state = {"supervisor_pid": os.getpid(), "created": psutil.Process().create_time(),
             "root": str(root), "config": str(config_path), "updated_at": utc(), "phase": "starting"}
    atomic_json(prior, state)
    budget_path=root / "budget.json"
    if not budget_path.exists():
        epoch=time.time()
        atomic_json(budget_path,{"started_epoch":epoch,"deadline_epoch":epoch+cfg.runtime_hours*3600,
                                "runtime_hours":cfg.runtime_hours,
                                "clock":"fixed wall-clock window including all phases and downtime"})
    budget=json.loads(budget_path.read_text(encoding="utf-8"))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    try:
        status_path = root / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
        if not args.evaluation_only and status.get("state") not in ("trained", "complete", "evaluation_incomplete"):
            with (root / "train.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-m", "az_gomoku.train", "--root", str(root),
                                          "--config", str(config_path)], cwd=ROOT, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
                state.update(phase="train", child_pid=child.pid, updated_at=utc())
                atomic_json(prior, state)
                code = child.wait()
            if code:
                raise RuntimeError(f"Training process failed with exit {code}; preserve checkpoint and failure cost")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        elapsed = status.get("elapsed_seconds", 0)
        remaining = budget["deadline_epoch"] - time.time()
        if remaining <= 0:
            raise RuntimeError("Runtime exhausted before final assessment")
        prior_assessment = json.loads((root / "assessment.json").read_text(encoding="utf-8")) if (root / "assessment.json").exists() else None
        if prior_assessment is None or prior_assessment.get("incomplete", True):
            state.update(phase="evaluation", child_pid=None, updated_at=utc())
            atomic_json(prior, state)
            from az_gomoku.evaluate import final_evaluate
            assessment = final_evaluate(root, cfg, time.monotonic() + remaining)
        else:
            assessment = json.loads((root / "assessment.json").read_text(encoding="utf-8"))
        status.update(state="evaluation_incomplete" if assessment["incomplete"] else "complete", phase="assessment", updated_at=utc(),
                      evaluation_incomplete=assessment["incomplete"], assessment=str(root / "assessment.json"))
        atomic_json(status_path, status)
        state.update(phase="evaluation_incomplete" if assessment["incomplete"] else "complete", updated_at=utc())
        atomic_json(prior, state)
    except BaseException as error:
        state.update(phase="failed", error=repr(error), traceback=traceback.format_exc(), updated_at=utc())
        atomic_json(prior, state)
        raise


if __name__ == "__main__":
    main()
