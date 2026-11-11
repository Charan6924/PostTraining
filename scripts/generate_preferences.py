"""Generate original-model DPO preference pairs.

Run from the repository root: uv run python scripts/generate_preferences.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Generate original-model DPO preference pairs.', 'configs/dpo.yaml')
