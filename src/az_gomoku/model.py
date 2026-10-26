"""Board Transformer with bidirectional attention, 2-D RoPE and policy/value heads."""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NetworkConfig:
    size: int = 15
    width: int = 128
    layers: int = 6
    heads: int = 4
    ff: int = 256
    # Missing in v1 checkpoints; retain their scalar head when loading them.
    # New training runs explicitly request the categorical W/L/D head.
    value_head: str = "scalar"

    def __post_init__(self):
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in (self.size, self.width, self.layers, self.heads, self.ff)):
            raise ValueError("Network dimensions and layer counts must be positive integers")
        if self.width % self.heads or (self.width // self.heads) % 4:
            raise ValueError("Each attention head must have width divisible by four")
        if self.value_head not in ("scalar", "wdl"):
            raise ValueError("value_head must be 'scalar' or 'wdl'")


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.dim = cfg.heads, cfg.width // cfg.heads
        self.qkv = nn.Linear(cfg.width, cfg.width * 3, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        freq = -math.log(10000) * torch.arange(self.dim // 4).float() / (self.dim // 4)
        self.log_frequency = nn.Parameter(freq.repeat(2, self.heads, 1))
        row, col = torch.meshgrid(torch.arange(cfg.size), torch.arange(cfg.size), indexing="ij")
        self.register_buffer("positions", torch.stack((row.flatten(), col.flatten())).float() - (cfg.size - 1) / 2,
                             persistent=False)

    def rotate(self, x):
        # x [batch, head, cell, dim], split half the channels per spatial axis.
        angle = self.positions[:, None, :, None] * self.log_frequency.exp()[:, :, None, :]
        cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        pieces = []
        for axis, part in enumerate(x.chunk(2, dim=-1)):
            pairs = part.reshape(*part.shape[:-1], self.dim // 4, 2)
            even, odd = pairs.unbind(-1)
            pieces.append(torch.stack((even * cos[axis] - odd * sin[axis],
                                       odd * cos[axis] + even * sin[axis]), -1).flatten(-2))
        return torch.cat(pieces, -1)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(self.rotate(q), self.rotate(k), v,
                                             dropout_p=0., is_causal=False)
        return self.out(out.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(cfg.width), nn.RMSNorm(cfg.width)
        self.attention = Attention(cfg)
        self.gate = nn.Linear(cfg.width, cfg.ff * 2, bias=False)
        self.down = nn.Linear(cfg.ff, cfg.width, bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        gate, up = self.gate(self.norm2(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class BoardTransformer(nn.Module):
    WDL_CLASSES = ("win", "loss", "draw")

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or NetworkConfig()
        self.stem = nn.Sequential(nn.Conv2d(4, self.cfg.width, 3, padding=1), nn.SiLU())
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(self.cfg.layers))
        self.norm = nn.RMSNorm(self.cfg.width)
        self.policy = nn.Linear(self.cfg.width, 1)
        value_outputs = 3 if self.cfg.value_head == "wdl" else 1
        # Keep the scalar checkpoint's value.0/value.2 parameter names and shapes.
        self.value = nn.Sequential(nn.Linear(self.cfg.width * 2, self.cfg.width), nn.SiLU(),
                                   nn.Linear(self.cfg.width, value_outputs),
                                   nn.Tanh() if self.cfg.value_head == "scalar" else nn.Identity())
        nn.init.normal_(self.policy.weight, std=.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.zeros_(self.value[-2].bias)

    def forward(self, planes, return_value_logits=False):
        """Return FP32 policy logits and current-player expected outcome.

        Training may additionally request FP32 pre-activation value logits:
        shape [B] for legacy scalar heads, or [B, 3] in win/loss/draw order.
        In particular, never round tanh's output to BF16 before its backward
        pass: at moderate logits that would turn nonzero gradients into zero.
        """
        x = self.stem(planes).flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        policy_logits = self.policy(x).squeeze(-1).float()
        features = torch.cat((x.mean(1), x.amax(1)), -1)
        value_logits = self.value[2](self.value[1](self.value[0](features))).float()
        with torch.autocast(device_type=planes.device.type, enabled=False):
            if self.cfg.value_head == "scalar":
                value_logits = value_logits.squeeze(-1)
                value = value_logits.tanh()
            else:
                probabilities = value_logits.softmax(-1)
                value = probabilities[:, 0] - probabilities[:, 1]
        if return_value_logits:
            return policy_logits, value, value_logits
        return policy_logits, value

    def manifest(self):
        return {"architecture": "board_transformer_2d_rope", **asdict(self.cfg),
                "parameters": sum(p.numel() for p in self.parameters()),
                "causal_attention": False, "output_heads": ["policy", "value"],
                "value_classes": list(self.WDL_CLASSES) if self.cfg.value_head == "wdl" else None}


def make_planes(boards, colors, last_moves, size=15, device="cpu"):
    boards = torch.as_tensor(boards, device=device).reshape(-1, size, size)
    colors = torch.as_tensor(colors, device=device).reshape(-1, 1, 1)
    current = boards * colors
    last_moves = torch.as_tensor(last_moves, device=device, dtype=torch.long).reshape(-1)
    last = torch.zeros_like(boards, dtype=torch.float32).flatten(1)
    last.scatter_(1, last_moves.clamp_min(0)[:, None], (last_moves >= 0).float()[:, None])
    black = (colors == 1).expand_as(boards)
    return torch.stack(((current == 1).float(), (current == -1).float(),
                        last.reshape_as(boards), black.float()), 1)


@torch.inference_mode()
def infer(model, boards, colors, last, device="cuda", amp=True):
    planes = make_planes(boards, colors, last, model.cfg.size, device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=amp and torch.device(device).type == "cuda"):
        logits, value = model(planes)
    occupied = torch.as_tensor(boards, device=device).reshape(logits.shape) != 0
    if bool(occupied.all(-1).any()):
        raise ValueError("Terminal full boards must never enter inference")
    probs = logits.masked_fill(occupied, -torch.inf).softmax(-1)
    return probs.cpu().numpy(), value.cpu().numpy()
