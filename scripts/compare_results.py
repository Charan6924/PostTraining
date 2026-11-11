"""Compare saved benchmark results and compute costs.

Run from the repository root: uv run python scripts/compare_results.py --help
"""

from posttraining.cli import scaffold_command


if __name__ == "__main__":
    scaffold_command('Compare saved benchmark results and compute costs.', None)
