# Lab-Game-2024

## Para correr:
python run_2048.py --mode agent --agent-module agent_ppo --agent-class PPOAgent --episodes 50

## Para correr (con visualización):
python run_2048.py --mode agent --agent-module agent_ppo --agent-class PPOAgent --episodes 10 --render --step-delay 0.05

## Modo de entrenamiento:
python train_ppo.py --total-steps 10_000_000 --beta-pbrs 0.005 --device cuda --resume

beta controla la influencia:
 * 0.001 aprende por si solo.
 * 0.005 mas peso a la heurística
 * 0.1 full expectimax
