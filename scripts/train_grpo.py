"""Train GRPO with correctness rewards.

Run from the repository root: uv run python scripts/train_grpo.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Train GRPO with correctness rewards.', 'configs/grpo.yaml')
