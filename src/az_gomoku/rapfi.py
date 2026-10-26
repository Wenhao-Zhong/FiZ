"""Bounded, evaluation-only Rapfi protocol adapter (100 ms per move)."""
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


class Rapfi:
    def __init__(self, executable, config, milliseconds=100):
        self.milliseconds = milliseconds
        self.proc = subprocess.Popen([str(executable), "--config", str(config), "--force-utf8"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        self.inbox = queue.Queue()
        self.transcript = []
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        self.send("START 15")
        self.wait(r"OK", 15)
        self.send(f"INFO timeout_turn {milliseconds}\nINFO timeout_match 10000000\nINFO rule 0")

    def _read(self):
        for line in self.proc.stdout:
            self.transcript.append(line.rstrip())
            self.inbox.put(line.rstrip())
        self.inbox.put(None)

    def send(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def wait(self, pattern, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.inbox.get(timeout=max(.01, deadline - time.monotonic()))
            if line is None:
                raise RuntimeError("Rapfi exited before replying")
            if line.startswith("ERROR"):
                raise RuntimeError(line)
            if re.fullmatch(pattern, line):
                return line
        raise TimeoutError("Rapfi timed out")

    def move(self, board, color):
        # Internal indices are row*15+col. Piskvork is x(col),y(row).
        # Rapfi reconstructs side-to-move from the sequence order, inserting
        # passes for non-alternating lists. Supply alternating actual colors,
        # ending with the opponent, rather than sorting occupied coordinates.
        black=[a for a,stone in enumerate(board) if stone==1]
        white=[a for a,stone in enumerate(board) if stone==-1]
        if len(black)-len(white) != (0 if color==1 else 1):
            raise ValueError("Board counts disagree with actual side to move")
        ordered=[]
        for i in range(len(black)):
            ordered.append((black[i],1))
            if i<len(white):
                ordered.append((white[i],-1))
        lines = [f"{a%15},{a//15},{1 if stone==color else 2}" for a,stone in ordered]
        self.send("BOARD\n" + "\n".join(lines) + "\nDONE")
        x, y = map(int, self.wait(r"\d+,\d+", 15).split(","))
        if not 0 <= x < 15 or not 0 <= y < 15 or board[y*15+x]:
            raise RuntimeError("Rapfi returned an illegal action")
        return y*15+x

    def close(self):
        try:
            self.send("END")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
            self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
