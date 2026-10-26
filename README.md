# FiZ

**FiZ (Five-in-a-row Zero)** learns 15x15 freestyle Gomoku from scratch through AlphaZero-style self-play. Five or more stones win; there are no forbidden moves or Swap2 rules.

The model is a 6-layer, 128-wide Transformer with 4 attention heads, 2-D RoPE, policy and WDL value heads, and **1,023,236 parameters**.

**GPU:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM).

Training: **24 hours**, seed 101, BF16, 64 self-play slots. Moves use 512 PUCT simulations 25% of the time and 128 otherwise; only full-search targets enter replay. No Rapfi teacher is used.

## Results

Each row contains 100 games against single-threaded Rapfi with a 100 ms thinking limit per move. Search budgets use the same 50 opening/seed pairs, with colors swapped.

| Set | PUCT | W/D/L | Black wins | White wins |
|---|---:|---:|---:|---:|
| Empty board | 512 | 40/0/60 | 40/50 | 0/50 |
| Empty board | 1024 | 50/0/50 | 50/50 | 0/50 |
| Empty board | 2048 | 50/0/50 | 50/50 | 0/50 |
| Four-ply openings | 512 | 40/0/60 | 34/50 | 6/50 |
| Four-ply openings | 1024 | 38/0/62 | 31/50 | 7/50 |
| Four-ply openings | 2048 | 39/0/61 | 32/50 | 7/50 |

[512 records](results/final/) · [1024 records](results/final_1024/) · [2048 records](results/final_2048/) · [Search comparison](results/search_comparison.json)

## Progress

One game per condition from the empty board. Click W/L to view its numbered board diagram; JSON and SVG are in the same folder.

| Checkpoint | 512 black | 512 white | 1024 black | 2048 black |
|---:|:---:|:---:|:---:|:---:|
| 1h | [L](results/timepoints/h01-v512-black/board.png) | [L](results/timepoints/h01-v512-white/board.png) | [L](results/timepoints/h01-v1024-black/board.png) | [L](results/timepoints/h01-v2048-black/board.png) |
| 2h | [L](results/timepoints/h02-v512-black/board.png) | [L](results/timepoints/h02-v512-white/board.png) | [L](results/timepoints/h02-v1024-black/board.png) | [L](results/timepoints/h02-v2048-black/board.png) |
| 3h | [L](results/timepoints/h03-v512-black/board.png) | [L](results/timepoints/h03-v512-white/board.png) | [L](results/timepoints/h03-v1024-black/board.png) | [L](results/timepoints/h03-v2048-black/board.png) |
| 4h | [L](results/timepoints/h04-v512-black/board.png) | [L](results/timepoints/h04-v512-white/board.png) | [L](results/timepoints/h04-v1024-black/board.png) | [L](results/timepoints/h04-v2048-black/board.png) |
| 8h | [L](results/timepoints/h08-v512-black/board.png) | [L](results/timepoints/h08-v512-white/board.png) | [W](results/timepoints/h08-v1024-black/board.png) | [L](results/timepoints/h08-v2048-black/board.png) |
| 12h | [L](results/timepoints/h12-v512-black/board.png) | [L](results/timepoints/h12-v512-white/board.png) | [L](results/timepoints/h12-v1024-black/board.png) | [L](results/timepoints/h12-v2048-black/board.png) |
| 16h | [L](results/timepoints/h16-v512-black/board.png) | [L](results/timepoints/h16-v512-white/board.png) | [L](results/timepoints/h16-v1024-black/board.png) | [W](results/timepoints/h16-v2048-black/board.png) |
| 20h | [W](results/timepoints/h20-v512-black/board.png) | [L](results/timepoints/h20-v512-white/board.png) | [L](results/timepoints/h20-v1024-black/board.png) | [L](results/timepoints/h20-v2048-black/board.png) |
| 24h | [L](results/timepoints/h24-v512-black/board.png) | [L](results/timepoints/h24-v512-white/board.png) | [W](results/timepoints/h24-v1024-black/board.png) | [W](results/timepoints/h24-v2048-black/board.png) |

## Run

Install PyTorch and a C++17 compiler. Install [Rapfi](https://github.com/dotb1ue/rapfi) and matching networks separately, then set `rapfi_executable`, `rapfi_config` and all weight files in `rapfi_assets` in `configs/transformer.json`. The Rapfi TOML must point to those weights; relative paths resolve from this checkout.

```sh
python -m pip install -e ".[dev,plots]"
python scripts/build_az_search.py
python -m pytest -q
python -m az_gomoku.train --root runs/example --config configs/transformer.json
python scripts/export_model.py --checkpoint runs/example/latest.pt --output runs/example/model.pt
python scripts/evaluate_rapfi.py --model models/final.pt --config configs/transformer.json --output runs/evaluation-2048 --hours 2 --simulations 2048
```

The preset trains for 24 hours; evaluation is separate. `--simulations` defaults to 512 and changes evaluation only. Use a separate output directory for each search budget.

Compact weights are in `models/`; load with `torch.load(..., weights_only=True)`. [Model manifest](models/manifest.json).

Repository license: pending.
