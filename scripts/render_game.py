"""Render a Gomoku game JSON as numbered PNG and SVG board diagrams."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from az_gomoku.rules import reference_replay


def winning_line(board, last):
    if last is None:
        return []
    row, col = divmod(last, 15)
    flat = board.ravel()
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        before, after = [], []
        for sign, target in ((-1, before), (1, after)):
            r, c = row + sign * dr, col + sign * dc
            while 0 <= r < 15 and 0 <= c < 15 and flat[r * 15 + c] == flat[last]:
                target.append(r * 15 + c)
                r, c = r + sign * dr, c + sign * dc
        line = before[::-1] + [last] + after
        if len(line) >= 5:
            return line
    return []


def validated_result(record):
    history = record["history"]
    if not isinstance(history, list) or any(type(action) is not int for action in history):
        raise ValueError("History must be a list of integer board indices")
    state = reference_replay(history)
    color = record.get("model_color", record.get("color", 1))
    if color not in (1, -1):
        raise ValueError("Model color must be +1 (black) or -1 (white)")
    complete = record.get("complete", state.terminated)
    if complete and not state.terminated:
        raise ValueError("A complete game must have a terminal board")
    winner = (0 if state.winner == -1 else 1 if state.winner == 0 else -1) if complete else None
    score = (.5 if winner == 0 else float(winner == color)) if complete else None
    if "score" in record and record["score"] != score:
        raise ValueError("Recorded score disagrees with the replayed board")
    line = winning_line(state.board, history[-1]) if complete and winner else []
    label = "INCOMPLETE" if not complete else "MODEL WIN" if score == 1 else "DRAW" if score == .5 else "OPPONENT WIN"
    return history, color, label, line


def render_game(record, output):
    """Use legal move history as the authority for stones and winning line."""
    history, color, _, line = validated_result(record)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10.8), dpi=180)
    fig.patch.set_facecolor("#faf9f6")
    ax.set_facecolor("#e6c795")
    ax.set_aspect("equal")
    for i in range(15):
        ax.plot([0, 14], [i, i], color="#796344", lw=.65, zorder=0)
        ax.plot([i, i], [0, 14], color="#796344", lw=.65, zorder=0)
    for x, y in ((3, 3), (11, 3), (7, 7), (3, 11), (11, 11)):
        ax.add_patch(Circle((x, y), .08, color="#5c4932", zorder=1))
    for move, action in enumerate(history, 1):
        row, col = divmod(action, 15)
        black = move % 2 == 1
        ax.add_patch(Circle((col, 14 - row), .43, facecolor="#1b1b1b" if black else "white",
                            edgecolor="#b52e31" if action in line else "#454545",
                            linewidth=2.4 if action in line else .8, zorder=3))
        ax.text(col, 14 - row, str(move), ha="center", va="center", color="white" if black else "#111111",
                fontsize=10 if move >= 100 else 12, weight="bold", zorder=4)
    ax.set(xlim=(-.65, 14.65), ylim=(-.65, 14.65), xticks=range(15), yticks=range(15))
    ax.set_xticklabels(list("ABCDEFGHIJKLMNO"))
    ax.set_yticklabels(range(1, 16))
    ax.tick_params(length=0, labelsize=11, pad=6)
    checkpoint = f"Checkpoint {record['slot_hour']}h" if "slot_hour" in record else "Gomoku game"
    model_name = record.get("model_name", "FiZ")
    opponent_name = record.get("opponent_name", "Rapfi")
    black, white = (model_name, opponent_name) if color == 1 else (opponent_name, model_name)
    fig.suptitle(f"{checkpoint}\nBlack: {black} | White: {white}", fontsize=16, weight="bold", y=.975)
    fig.subplots_adjust(left=.07, right=.97, bottom=.055, top=.875)
    paths = []
    for extension in ("png", "svg"):
        path = output / f"board.{extension}"
        fig.savefig(path, facecolor=fig.get_facecolor())
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Directory for board.png and board.svg")
    args = parser.parse_args()
    for path in render_game(json.loads(args.game.read_text(encoding="utf-8")), args.output):
        print(path)


if __name__ == "__main__":
    main()
