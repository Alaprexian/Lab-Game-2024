# Lab-Game-2024

## Para correr:
python run_2048.py --mode agent --agent-module agent_ppo --agent-class PPOAgent --episodes 50

## Para correr (con visualización):
python run_2048.py --mode agent --agent-module agent_ppo --agent-class PPOAgent --episodes 10 --render --step-delay 0.05

## Modo de entrenamiento:
python train_ppo.py --total-steps 10_000_000 --beta-pbrs 0.005 --device cuda --resume
