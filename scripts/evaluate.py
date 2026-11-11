"""Evaluate a baseline or trained checkpoint.

Run from the repository root: uv run python scripts/evaluate.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Evaluate a baseline or trained checkpoint.', 'configs/eval.yaml')
