# train_ppo.py
"""
Entrenamiento PPO para 2048 con Potential-Based Reward Shaping (PBRS).

La recompensa moldeada es:
    r_shaped = r_game + β · (Φ(s') − Φ(s))

donde Φ(s) es la heurística del agente expectimax:
    Φ(s) = pesos_posicionales · tablero
          + w_empty · (celdas_vacías²)
          − w_corner  (si max tile no está en [0,0])
          + w_monotone · monotonía(tablero)
          − w_smooth  · suavidad(tablero)

Esta forma garantiza que la política óptima del MDP moldeado
coincide con la del MDP original (Ng et al., 1999).

Uso:
    python train_ppo.py                           # defaults (3M pasos)
    python train_ppo.py --total-steps 10_000_000  # entrenamiento largo
    python train_ppo.py --device cpu              # sin GPU
    python train_ppo.py --beta-pbrs 0.005         # más influencia heurística

Resultado: ppo_2048.pt

Evaluar:
    python run_2048.py --mode agent \\
        --agent-module agent_ppo --agent-class PPOAgent \\
        --episodes 50
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_2048 import Game2048
from agent_ppo import PPONet, encode_board, ACTIONS, ACTION_TO_IDX, N_LEVELS


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Heurística  Φ(s)  — basada en el agente expectimax
# ══════════════════════════════════════════════════════════════════════════════

class Heuristic:
    """
    Potencial Φ(s) que combina las heurísticas del agente expectimax
    con dos términos adicionales (monotonía y suavidad).

    Componentes:
      1. Alineación posicional   – tiles grandes en esquina sup-izq.
      2. Bonus de celdas vacías  – mayor espacio → mejor supervivencia.
      3. Penalización de esquina – castiga si el tile máximo no está en [0,0].
      4. Monotonía               – filas/columnas ordenadas (↑ es mejor).
      5. Suavidad                – penaliza diferencias grandes entre vecinos.
    """

    # Matriz de pesos idéntica al agente expectimax original
    WEIGHTS = np.array([
        [100, 80, 70, 60],
        [ 10, 20, 30, 40],
        [  9,  8,  7,  6],
        [  2,  3,  4,  5],
    ], dtype=np.float32)

    def __init__(
        self,
        w_position:  float = 1.0,
        w_empty:     float = 10.0,
        w_corner:    float = 500.0,
        w_monotone:  float = 20.0,
        w_smooth:    float = 5.0,
    ):
        self.w_position = w_position
        self.w_empty    = w_empty
        self.w_corner   = w_corner
        self.w_monotone = w_monotone
        self.w_smooth   = w_smooth

    def __call__(self, board: np.ndarray) -> float:
        return self._phi(board)

    def _phi(self, board: np.ndarray) -> float:
        if np.max(board) == 0:
            return 0.0

        phi = 0.0

        # 1. Alineación posicional  (igual al agente original)
        phi += self.w_position * float(np.sum(board * self.WEIGHTS))

        # 2. Bonus cuadrático de celdas vacías  (igual al agente original)
        empty = int(np.count_nonzero(board == 0))
        phi  += self.w_empty * (empty ** 2)

        # 3. Penalización si el tile máximo no está en la esquina  (igual al original)
        if board[0, 0] != np.max(board):
            phi -= self.w_corner

        # 4. Monotonía: secuencias ordenadas en filas y columnas
        phi += self.w_monotone * self._monotonicity(board)

        # 5. Suavidad: penaliza tiles adyacentes muy distintos
        phi -= self.w_smooth * self._smoothness(board)

        return float(phi)

    # ── Componentes auxiliares ────────────────────────────────────────────────

    @staticmethod
    def _monotonicity(board: np.ndarray) -> float:
        """
        Para cada fila y columna, toma el máximo entre la
        tendencia creciente y decreciente (en escala log2).
        """
        score = 0.0
        size  = board.shape[0]

        def _l(v):
            return float(np.log2(v)) if v > 0 else 0.0

        for i in range(size):
            row = [_l(board[i, j]) for j in range(size)]
            col = [_l(board[j, i]) for j in range(size)]
            for seq in (row, col):
                inc = dec = 0.0
                for k in range(size - 1):
                    d = seq[k + 1] - seq[k]
                    if d > 0:
                        inc += d
                    else:
                        dec -= d
                score += max(inc, dec)
        return score

    @staticmethod
    def _smoothness(board: np.ndarray) -> float:
        """
        Suma de |log2(a) − log2(b)| para pares de tiles adyacentes no vacíos.
        """
        total = 0.0
        size  = board.shape[0]
        for r in range(size):
            for c in range(size):
                v = board[r, c]
                if v == 0:
                    continue
                lv = np.log2(v)
                for dr, dc in ((0, 1), (1, 0)):
                    nr, nc = r + dr, c + dc
                    if nr < size and nc < size and board[nr, nc] > 0:
                        total += abs(lv - np.log2(board[nr, nc]))
        return total


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Envoltorio PBRS
# ══════════════════════════════════════════════════════════════════════════════

class PBRSReward:
    """
    Aplica Potential-Based Reward Shaping:

        r_shaped(s, a, s') = r_game  +  β · (Φ(s') − Φ(s))

    Propiedades:
      • Si β es pequeño (≤ 0.01), la señal de juego sigue dominando.
      • El shaping NO cambia la política óptima (Ng et al., 1999).
      • El gradiente de Φ guía al agente hacia posiciones estructuralmente
        mejores incluso cuando r_game es 0 (movimientos sin fusión).
    """

    def __init__(self, heuristic: Heuristic, beta: float = 0.001):
        self.phi  = heuristic
        self.beta = beta
        self._phi_prev: float = 0.0

    def reset(self, board: np.ndarray) -> None:
        """Llama al inicio de cada episodio."""
        self._phi_prev = self.phi(board)

    def shaped(self, r_game: float, next_board: np.ndarray, done: bool) -> float:
        """
        Calcula r_shaped para una transición (s → s').
        Si done=True, Φ(s') = 0 (convención de estado terminal).
        """
        phi_next = 0.0 if done else self.phi(next_board)
        r_shaped = r_game + self.beta * (phi_next - self._phi_prev)
        self._phi_prev = phi_next
        return r_shaped


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Utilidades GAE
# ══════════════════════════════════════════════════════════════════════════════

def make_legal_mask(legal_actions: List[str], device) -> torch.Tensor:
    mask = torch.zeros(4, dtype=torch.bool, device=device)
    for a in legal_actions:
        mask[ACTION_TO_IDX[a]] = True
    return mask


def compute_gae(
    rewards:    torch.Tensor,
    values:     torch.Tensor,
    next_value: float,
    dones:      torch.Tensor,
    gamma:      float,
    lam:        float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae    = 0.0
    v_next = next_value

    for t in reversed(range(T)):
        nterm = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * v_next * nterm - values[t]
        gae   = delta + gamma * lam * nterm * gae
        advantages[t] = gae
        v_next = values[t]

    return advantages, advantages + values


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Bucle PPO
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    device = torch.device(args.device)
    print(f"[train] Dispositivo       : {device}")
    print(f"[train] Pasos totales     : {args.total_steps:,}")
    print(f"[train] β PBRS            : {args.beta_pbrs}")
    print(f"[train] Pesos heurística  : "
          f"pos={args.w_position}, empty={args.w_empty}, "
          f"corner={args.w_corner}, mono={args.w_monotone}, "
          f"smooth={args.w_smooth}")

    # ── Entorno ───────────────────────────────────────────────────────────────
    game      = Game2048(size=args.board_size)
    heuristic = Heuristic(
        w_position = args.w_position,
        w_empty    = args.w_empty,
        w_corner   = args.w_corner,
        w_monotone = args.w_monotone,
        w_smooth   = args.w_smooth,
    )
    pbrs = PBRSReward(heuristic, beta=args.beta_pbrs)

    # ── Red ───────────────────────────────────────────────────────────────────
    net       = PPONet(board_size=args.board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[train] Parámetros red    : {sum(p.numel() for p in net.parameters()):,}")

    # ── Buffers ───────────────────────────────────────────────────────────────
    B = args.n_steps
    buf_obs      = torch.zeros(B, N_LEVELS, args.board_size, args.board_size)
    buf_actions  = torch.zeros(B, dtype=torch.long)
    buf_logprobs = torch.zeros(B)
    buf_rewards  = torch.zeros(B)        # recompensas MOLDEADAS y normalizadas
    buf_dones    = torch.zeros(B, dtype=torch.bool)
    buf_values   = torch.zeros(B)
    buf_masks    = torch.zeros(B, 4, dtype=torch.bool)

    # ── Métricas ──────────────────────────────────────────────────────────────
    ep_shaped:   deque = deque(maxlen=100)   # score con shaping (debug)
    ep_game:     deque = deque(maxlen=100)   # score real del juego
    ep_tiles:    deque = deque(maxlen=100)   # max tile por episodio
    acc_shaped = acc_game = 0.0

    # Normalización de recompensas (online)
    rw_mean = 0.0; rw_var = 1.0; rw_n = 0

    def normalize(r: float) -> float:
        nonlocal rw_mean, rw_var, rw_n
        rw_n += 1
        α = 1.0 / rw_n if rw_n < 100 else 0.01
        rw_mean += α * (r - rw_mean)
        rw_var  += α * ((r - rw_mean) ** 2 - rw_var)
        return r / max(rw_var ** 0.5, 1e-8)

    # ── Estado inicial ────────────────────────────────────────────────────────
    obs   = game.reset()
    legal = game.legal_actions()
    pbrs.reset(obs)

    t_start       = time.time()
    total_updates = args.total_steps // B
    sep           = "─" * 110
    print(sep)

    for update in range(1, total_updates + 1):

        # LR annealing
        optimizer.param_groups[0]["lr"] = args.lr * (1.0 - (update - 1) / total_updates)

        # ── Rollout ───────────────────────────────────────────────────────────
        for step in range(B):
            obs_t  = encode_board(obs)
            mask_t = make_legal_mask(legal, device)
            buf_obs[step]   = obs_t
            buf_masks[step] = mask_t

            with torch.no_grad():
                a_t, lp_t, _, v_t = net.get_action_and_value(
                    obs_t.unsqueeze(0).to(device),
                    legal_mask=mask_t.unsqueeze(0),
                )
            action_str = ACTIONS[a_t.item()]
            buf_actions[step]  = a_t.cpu()
            buf_logprobs[step] = lp_t.cpu()
            buf_values[step]   = v_t.cpu()

            result = game.step(action_str)
            r_game = float(result.reward) if result.info.get("moved", False) else 0.0

            # ── PBRS: r_shaped = r_game + β·(Φ(s') − Φ(s)) ──────────────────
            r_shaped = pbrs.shaped(r_game, result.obs, result.done)
            buf_rewards[step] = normalize(r_shaped)
            buf_dones[step]   = result.done

            acc_shaped += r_shaped
            acc_game   += r_game

            if result.done or not result.info["legal_actions"]:
                ep_shaped.append(acc_shaped)
                ep_game.append(acc_game)
                ep_tiles.append(int(game.board.max()))
                acc_shaped = acc_game = 0.0
                obs   = game.reset()
                legal = game.legal_actions()
                pbrs.reset(obs)
            else:
                obs   = result.obs
                legal = result.info["legal_actions"]

        # ── GAE ───────────────────────────────────────────────────────────────
        with torch.no_grad():
            _, nv = net(encode_board(obs).unsqueeze(0).to(device))
        adv, ret = compute_gae(buf_rewards, buf_values, nv.item(),
                               buf_dones, args.gamma, args.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ── Actualización PPO ─────────────────────────────────────────────────
        b_obs  = buf_obs.to(device);   b_act  = buf_actions.to(device)
        b_lp   = buf_logprobs.to(device); b_adv = adv.to(device)
        b_ret  = ret.to(device);       b_mask = buf_masks.to(device)

        idx_arr    = np.arange(B)
        pg_s = v_s = ent_s = clip_s = 0.0
        nb = 0

        for _ in range(args.n_epochs):
            np.random.shuffle(idx_arr)
            for start in range(0, B, args.batch_size):
                idx = idx_arr[start : start + args.batch_size]
                _, nlp, ent, nv = net.get_action_and_value(
                    b_obs[idx], action=b_act[idx], legal_mask=b_mask[idx]
                )
                ratio = (nlp - b_lp[idx]).exp()

                with torch.no_grad():
                    clip_s += ((ratio - 1).abs() > args.clip_eps).float().mean().item()

                a = b_adv[idx]
                pg  = torch.max(-a * ratio,
                                -a * ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps)).mean()
                vl  = 0.5 * (nv - b_ret[idx]).pow(2).mean()
                el  = ent.mean()
                loss = pg + args.vf_coef * vl - args.ent_coef * el

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_s += pg.item(); v_s += vl.item(); ent_s += el.item()
                nb   += 1

        # ── Log ───────────────────────────────────────────────────────────────
        if update % args.log_interval == 0 or update == total_updates:
            nb  = max(nb, 1)
            sps = update * B / (time.time() - t_start)
            msg = (
                f"Upd {update:5d}/{total_updates} | SPS {sps:6.0f} | "
                f"Score_game {np.mean(ep_game) if ep_game else 0:8.1f} | "
                f"Score_shaped {np.mean(ep_shaped) if ep_shaped else 0:10.1f} | "
                f"MaxTile(μ) {np.mean(ep_tiles) if ep_tiles else 0:5.0f} | "
                f"MaxTile(↑) {max(ep_tiles) if ep_tiles else 0:5d} | "
                f"PG {pg_s/nb:.4f} | V {v_s/nb:.4f} | "
                f"Ent {ent_s/nb:.3f} | Clip {clip_s/nb:.3f}"
            )
            print(msg)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if update % args.save_interval == 0 or update == total_updates:
            torch.save({
                "model_state_dict":     net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "steps":  update * B,
                "mean_score_game": float(np.mean(ep_game)) if ep_game else 0.0,
                "heuristic_config": vars(args),
            }, args.save_path)
            print(f"  → Checkpoint guardado: '{args.save_path}'")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("[train] ¡Entrenamiento completo!")
    print(f"  Modelo           : {args.save_path}")
    if ep_game:
        print(f"  Score juego (μ)  : {np.mean(ep_game):.1f}")
    if ep_tiles:
        print(f"  Max tile (μ)     : {np.mean(ep_tiles):.0f}")
        print(f"  Max tile (↑)     : {max(ep_tiles)}")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="PPO + PBRS para 2048")

    p.add_argument("--board-size",    type=int,   default=4)
    p.add_argument("--device",        type=str,   default="auto")
    p.add_argument("--total-steps",   type=int,   default=3_000_000)

    # PPO
    p.add_argument("--n-steps",       type=int,   default=2048)
    p.add_argument("--n-epochs",      type=int,   default=4)
    p.add_argument("--batch-size",    type=int,   default=512)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--gae-lambda",    type=float, default=0.95)
    p.add_argument("--clip-eps",      type=float, default=0.2)
    p.add_argument("--vf-coef",       type=float, default=0.5)
    p.add_argument("--ent-coef",      type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # ── Heurística (Φ) ── mismos valores por defecto que el agente expectimax
    p.add_argument("--w-position",  type=float, default=1.0,
                   help="Peso: alineación posicional (WEIGHTS · board)")
    p.add_argument("--w-empty",     type=float, default=10.0,
                   help="Peso: bonus celdas vacías  (empty_count²)")
    p.add_argument("--w-corner",    type=float, default=500.0,
                   help="Penalización si max tile ∉ [0,0]")
    p.add_argument("--w-monotone",  type=float, default=20.0,
                   help="Peso: bonus de monotonía en filas/columnas")
    p.add_argument("--w-smooth",    type=float, default=5.0,
                   help="Peso: penalización de suavidad (diferencia log2 entre vecinos)")

    # ── PBRS ──
    p.add_argument("--beta-pbrs",   type=float, default=0.001,
                   help="Escala del shaping: r_shaped = r_game + β·ΔΦ\n"
                        "0.001 = sutil | 0.005 = moderado | 0.01 = fuerte")

    # I/O
    p.add_argument("--save-path",    type=str, default="ppo_2048.pt")
    p.add_argument("--save-interval",type=int, default=200)
    p.add_argument("--log-interval", type=int, default=20)

    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


if __name__ == "__main__":
    train(parse_args())