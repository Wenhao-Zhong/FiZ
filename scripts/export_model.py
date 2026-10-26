"""Export a trusted local training checkpoint as a compact inference artifact."""
import argparse
from dataclasses import asdict
import hashlib
import io
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from az_gomoku.model import BoardTransformer, NetworkConfig


def export_model(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if checkpoint == output:
        raise ValueError("Output must differ from the training checkpoint")
    if output.exists():
        raise FileExistsError(output)
    # Hash and decode the same bytes. Full training checkpoints contain trusted
    # Python/NumPy optimizer and replay state; never use an untrusted input here.
    content = checkpoint.read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    source = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    network = NetworkConfig(**source["network"])
    model = BoardTransformer(network)
    model.load_state_dict(source["model"], strict=True)
    weights = {name: value.detach().cpu().contiguous().clone()
               for name, value in model.state_dict().items()}
    elapsed = float(source["elapsed_seconds"])
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid elapsed time")
    counters = {public: int(source[internal]) for public, internal in (
        ("real_plies", "plies"), ("completed_games", "games"),
        ("optimizer_steps", "steps"), ("selfplay_waves", "wave"))}
    search = {name: int(source["search"][name])
              for name in ("network_evaluations", "simulations")}
    if any(value < 0 for value in (*counters.values(), *search.values())):
        raise ValueError("Counters cannot be negative")
    counters["search"] = search
    counters["elapsed_seconds"] = elapsed
    artifact = {
        "format": "az-gomoku-model-v1",
        "config": asdict(network),
        "model": weights,
        "training": counters,
        "provenance": {
            "checkpoint_sha256": source_hash,
            "training_protocol_version": int(source["protocol_version"]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects an existing published artifact from overwrite.
    with output.open("xb") as stream:
        torch.save(artifact, stream)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Trusted full checkpoint from a completed local run")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    artifact = export_model(args.checkpoint, args.output)
    print(f"Exported {args.output}; source SHA256: {artifact['provenance']['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
