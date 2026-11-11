"""Prepare fixed GSM8K training/validation splits.

Run from the repository root: uv run python scripts/prepare_data.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Prepare fixed GSM8K training/validation splits.', None)
