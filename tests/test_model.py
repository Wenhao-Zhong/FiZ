"""CPU regressions for value precision, W/L/D learning and v1 compatibility."""
from dataclasses import asdict

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from az_gomoku.model import BoardTransformer, NetworkConfig, infer


def small_config(**kwargs):
    return NetworkConfig(**{ "size": 5, "width": 16, "layers": 1,
                            "heads": 2, "ff": 32, **kwargs})


def fixed_value_model(kind, biases):
    model = BoardTransformer(small_config(value_head=kind)).eval()
    with torch.no_grad():
        model.value[2].weight.zero_()
        model.value[2].bias.copy_(torch.tensor(biases))
    return model


def test_scalar_bf16_saturation_does_not_erase_value_gradient():
    # The v1 implementation rounded tanh(-3.5) to BF16 -1, giving zero
    # derivative even when the correct game outcome was the opposite (+1).
    legacy_input = torch.tensor([-3.5], dtype=torch.bfloat16, requires_grad=True)
    legacy_value = legacy_input.tanh().float()
    (legacy_value - 1).square().sum().backward()
    assert legacy_value.item() == -1 and legacy_input.grad.item() == 0

    model = fixed_value_model("scalar", [-3.5])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        policy, value, logits = model(torch.zeros(1, 4, 5, 5), return_value_logits=True)
        loss = F.mse_loss(value, torch.ones_like(value))
    loss.backward()
    assert policy.dtype == value.dtype == logits.dtype == torch.float32
    assert logits.shape == (1,)
    assert -1 < value.item() < -.99
    assert torch.isfinite(model.value[2].bias.grad).all()
    assert model.value[2].bias.grad.item() < -.005


def test_wdl_extremely_wrong_prediction_has_corrective_cross_entropy_gradient():
    model = fixed_value_model("wdl", [-20., 20., 0.])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        policy, value, logits = model(torch.zeros(1, 4, 5, 5), return_value_logits=True)
        # A win is class 0. Classification must still learn when expected
        # outcome is numerically -1, unlike differentiating through tanh/MSE.
        loss = F.cross_entropy(logits, torch.tensor([0]))
    loss.backward()
    assert policy.dtype == value.dtype == logits.dtype == torch.float32
    assert logits.shape == (1, 3)
    assert value.item() == -1
    grad = model.value[2].bias.grad
    assert grad[0] < -.9 and grad[1] > .9 and torch.isfinite(grad).all()


def test_wdl_class_order_and_default_forward_contract():
    for biases, expected in (([20., -20., 0.], 1.),
                             ([-20., 20., 0.], -1.),
                             ([0., 0., 20.], 0.)):
        model = fixed_value_model("wdl", biases)
        with torch.no_grad():
            outputs = model(torch.zeros(2, 4, 5, 5))
        assert len(outputs) == 2
        torch.testing.assert_close(outputs[1], torch.full((2,), expected))
    assert model.manifest()["value_classes"] == ["win", "loss", "draw"]


def test_legacy_checkpoint_without_value_head_still_loads_strictly(tmp_path):
    torch.manual_seed(17)
    original = BoardTransformer(small_config()).eval()
    old_config = asdict(original.cfg)
    old_config.pop("value_head")
    # These are the original v1 head parameter names and dimensions.
    assert original.state_dict()["value.0.weight"].shape == (16, 32)
    assert original.state_dict()["value.2.weight"].shape == (1, 16)
    path = tmp_path / "legacy.pt"
    torch.save({"config": old_config, "model": original.state_dict()}, path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = BoardTransformer(NetworkConfig(**checkpoint["config"])).eval()
    restored.load_state_dict(checkpoint["model"], strict=True)
    assert restored.cfg.value_head == "scalar"
    x = torch.randn(2, 4, 5, 5)
    with torch.no_grad():
        expected = original(x)
        actual = restored(x)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["scalar", "wdl"])
def test_inference_mask_probabilities_and_terminal_rejection(kind):
    model = BoardTransformer(small_config(value_head=kind)).eval()
    boards = np.zeros((2, 25), np.int8)
    boards[:, 6] = 1
    boards[:, 12] = -1
    policy, value = infer(model, boards, [-1, 1], [12, 12], device="cpu", amp=False)
    assert policy.dtype == value.dtype == np.float32
    assert policy.shape == (2, 25) and value.shape == (2,)
    assert not policy[:, [6, 12]].any()
    np.testing.assert_allclose(policy.sum(-1), 1, atol=1e-6)
    assert np.isfinite(policy).all() and np.isfinite(value).all()
    assert (np.abs(value) <= 1).all()
    with pytest.raises(ValueError, match="Terminal full boards"):
        infer(model, np.ones((1, 25), np.int8), [1], [24], device="cpu", amp=False)


@pytest.mark.parametrize("kwargs", [
    {"heads": 0}, {"layers": 0}, {"size": -1}, {"ff": 0}, {"width": 15},
    {"width": 12, "heads": 2}, {"value_head": "unknown"}, {"layers": True},
])
def test_invalid_network_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        small_config(**kwargs)
