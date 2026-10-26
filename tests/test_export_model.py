import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from az_gomoku.model import BoardTransformer, NetworkConfig


def exporter():
    path = Path(__file__).resolve().parents[1] / "scripts/export_model.py"
    spec = importlib.util.spec_from_file_location("export_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.export_model


def checkpoint(tmp_path):
    network = dict(size=15, width=16, layers=1, heads=2, ff=32, value_head="wdl")
    model = BoardTransformer(NetworkConfig(**network)).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save({"network": network, "model": model.state_dict(),
                "protocol_version": 2, "plies": 23, "games": 1, "steps": 2,
                "wave": 8, "elapsed_seconds": 12.5,
                "search": {"network_evaluations": 99, "simulations": 120},
                "optimizer": {"private_state": np.zeros(32)},
                "replay": np.ones(1000), "numpy_rng": np.random.default_rng(1).bit_generator.state}, path)
    return path, model


def test_export_is_compact_safe_and_matches_original(tmp_path):
    source, original = checkpoint(tmp_path)
    output = tmp_path / "models/final.pt"
    exporter()(source, output)
    artifact = torch.load(output, map_location="cpu", weights_only=True)
    assert set(artifact) == {"format", "config", "model", "training", "provenance"}
    assert artifact["format"] == "az-gomoku-model-v1"
    assert artifact["provenance"]["checkpoint_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert artifact["training"]["real_plies"] == 23
    assert artifact["training"]["search"]["simulations"] == 120
    restored = BoardTransformer(NetworkConfig(**artifact["config"])).eval()
    restored.load_state_dict(artifact["model"], strict=True)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    assert all(value.device.type == "cpu" for value in artifact["model"].values())
    assert output.stat().st_size < source.stat().st_size


def test_export_refuses_overwrite_or_input_replacement(tmp_path):
    source, _ = checkpoint(tmp_path)
    export = exporter()
    original_bytes = source.read_bytes()
    with pytest.raises(ValueError, match="differ"):
        export(source, source)
    output = tmp_path / "final.pt"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        export(source, output)
    assert source.read_bytes() == original_bytes
    assert output.read_bytes() == b"existing"


def test_invalid_model_is_not_published(tmp_path):
    source, _ = checkpoint(tmp_path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    payload["model"].pop(next(iter(payload["model"])))
    torch.save(payload, source)
    output = tmp_path / "final.pt"
    with pytest.raises(RuntimeError):
        exporter()(source, output)
    assert not output.exists()
