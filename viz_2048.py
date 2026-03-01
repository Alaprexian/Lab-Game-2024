# viz_2048.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


_BG = "#bbada0"
_EMPTY = "#cdc1b4"
_TEXT_DARK = "#776e65"
_TEXT_LIGHT = "#f9f6f2"

_TILE_COLORS = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
}


def _tile_color(v: int) -> str:
    if v <= 0:
        return _EMPTY
    return _TILE_COLORS.get(v, "#3c3a32")


def _text_color(v: int) -> str:
    if v <= 0:
        return _TEXT_DARK
    return _TEXT_DARK if v in (2, 4) else _TEXT_LIGHT


@dataclass
class Renderer2048:
    size: int
    fig: plt.Figure
    ax: plt.Axes

    @classmethod
    def create(cls, size: int, window_title: str = "2048") -> "Renderer2048":
        fig, ax = plt.subplots(figsize=(5.2, 5.8))
        try:
            fig.canvas.manager.set_window_title(window_title)
        except Exception:
            pass

        ax.set_aspect("equal")
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.axis("off")

        fig.patch.set_facecolor(_BG)
        ax.set_facecolor(_BG)

        fig.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)

        return cls(size=size, fig=fig, ax=ax)

    def draw(self, board: np.ndarray, score: int = 0, status: str = "") -> None:
        if board.shape != (self.size, self.size):
            raise ValueError(f"board must have shape {(self.size, self.size)}")

        self.ax.clear()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        self.ax.axis("off")
        self.fig.patch.set_facecolor(_BG)
        self.ax.set_facecolor(_BG)

        for t in list(self.fig.texts):
            t.remove()

        header = f"Score: {score}"
        if status:
            header += f"   |   {status}"

        self.fig.text(
            0.5,
            0.94, 
            header,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=_TEXT_LIGHT,
        )

        pad = 0.08
        rounding = 0.12

        for r in range(self.size):
            for c in range(self.size):
                v = int(board[r, c])
                x = c
                y = self.size - 1 - r

                rect = FancyBboxPatch(
                    (x + pad, y + pad),
                    1 - 2 * pad,
                    1 - 2 * pad,
                    boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                    linewidth=0,
                    facecolor=_tile_color(v),
                )
                self.ax.add_patch(rect)

                if v > 0:
                    digits = len(str(v))
                    fs = 26 if digits <= 2 else 22 if digits == 3 else 18 if digits == 4 else 14
                    self.ax.text(
                        x + 0.5,
                        y + 0.5,
                        str(v),
                        ha="center",
                        va="center",
                        fontsize=fs,
                        fontweight="bold",
                        color=_text_color(v),
                    )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
