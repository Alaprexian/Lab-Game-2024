# run_2048.py
from __future__ import annotations

import argparse
import importlib
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from game_2048 import Game2048 
from viz_2048 import Renderer2048


def _load_agent(agent_module: str, agent_class: str, seed: Optional[int]):
    mod = importlib.import_module(agent_module)
    cls = getattr(mod, agent_class)
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()


def run_manual(size: int, seed: Optional[int]) -> None:
    game = Game2048(size=size, seed=seed)
    renderer = Renderer2048.create(size=size, window_title="2048 (manual)")

    score = 0
    status = "Use arrows / WASD. R=reset, Q=quit."

    renderer.draw(game.board, score=score, status=status)

    def on_key(event):
        nonlocal score, status, game

        if event.key is None:
            return

        k = event.key.lower()

        if k in ("q", "escape"):
            plt.close(renderer.fig)
            return

        if k == "r":
            game.reset()
            score = 0
            status = "Reset. Use arrows / WASD. R=reset, Q=quit."
            renderer.draw(game.board, score=score, status=status)
            return

        keymap = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "w": "up",
            "s": "down",
            "a": "left",
            "d": "right",
        }

        if k not in keymap:
            return

        action = keymap[k]
        result = game.step(action)

        if result.info.get("moved", False):
            score += result.reward

        if result.done:
            status = f"Game over. Final score={score}. Press R to restart."
        else:
            status = "Use arrows / WASD. R=reset, Q=quit."

        renderer.draw(result.obs, score=score, status=status)

    renderer.fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def run_agent(
    size: int,
    seed: Optional[int],
    agent_module: str,
    agent_class: str,
    episodes: int,
    render: bool,
    max_steps: int,
    step_delay: float,
) -> None:
    agent = _load_agent(agent_module, agent_class, seed=seed)

    renderer = None
    if render:
        renderer = Renderer2048.create(size=size, window_title="2048 (agent)")
        plt.ion()

    scores = []
    max_tiles = []

    for ep in range(1, episodes + 1):
        game = Game2048(size=size, seed=None if seed is None else (seed + ep))
        score = 0

        if renderer is not None:
            renderer.draw(game.board, score=score, status=f"Episode {ep}/{episodes}")

        steps = 0
        while True:
            legal = game.legal_actions()
            if not legal:
                break

            # State:
            # - game.board: np.ndarray of shape (size, size) with current board state
            #   Example:
            #     [[   0,   2,   4,   0]
            #      [   2,   4,   8,   0]
            #      [   0,   2,  16,  32]
            #      [   0,   2,   2,  16]]
            # Actions:
            # - action: str in {"up", "down", "left", "right"}

            # Optional (for Random Agent):
            # - legal: List[str] of legal actions, each in {"up", "down", "left", "right"}
            action = agent.act(game.board.copy(), legal)
            result = game.step(action)

            if result.info.get("moved", False):
                score += result.reward

            steps += 1
            if renderer is not None:
                renderer.draw(result.obs, score=score, status=f"Ep {ep}/{episodes} | step {steps}")
                plt.pause(step_delay)

            if result.done or steps >= max_steps:
                break

        scores.append(score)
        max_tiles.append(int(game.board.max()))

        if renderer is not None:
            renderer.draw(game.board, score=score, status=f"Episode {ep} done. max_tile={max_tiles[-1]}")
            plt.pause(0.25)

    if renderer is not None:
        plt.ioff()
        plt.show(block=False)

    scores_arr = np.array(scores, dtype=float)
    max_tiles_arr = np.array(max_tiles, dtype=int)

    print("=== Results ===")
    print(f"Episodes: {episodes}")
    print(f"Mean score: {scores_arr.mean():.2f}   Std: {scores_arr.std(ddof=1) if episodes > 1 else 0.0:.2f}")
    print(f"Median score: {np.median(scores_arr):.2f}")
    print(f"Mean max-tile: {max_tiles_arr.mean():.2f}")
    print(f"Max of max-tile: {max_tiles_arr.max()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--mode", type=str, choices=("manual", "agent"), default="manual")

    # agent mode options
    p.add_argument("--agent-module", type=str, default="agent_random")
    p.add_argument("--agent-class", type=str, default="RandomAgent")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--step-delay", type=float, default=0.03)

    args = p.parse_args()

    if args.mode == "manual":
        run_manual(size=args.size, seed=args.seed)
    else:
        run_agent(
            size=args.size,
            seed=args.seed,
            agent_module=args.agent_module,
            agent_class=args.agent_class,
            episodes=args.episodes,
            render=args.render,
            max_steps=args.max_steps,
            step_delay=args.step_delay,
        )


if __name__ == "__main__":
    main()
