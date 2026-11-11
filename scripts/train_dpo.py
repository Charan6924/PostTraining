"""Train DPO without an SFT stage.

Run from the repository root: uv run python scripts/train_dpo.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Train DPO without an SFT stage.', 'configs/dpo.yaml')
