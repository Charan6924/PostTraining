"""Train PPO with correctness rewards and a value estimator.

Run from the repository root: uv run python scripts/train_ppo.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Train PPO with correctness rewards and a value estimator.', 'configs/ppo.yaml')
