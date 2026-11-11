"""Skeleton commands; --help works without loading models or datasets."""

import argparse


def scaffold_command(description, default_config=None):
    """Describe a planned command without pretending it is implemented."""
    parser = argparse.ArgumentParser(description=description)
    if default_config:
        parser.add_argument("--config", default=default_config)
    else:
        parser.add_argument("--config", required=True)
    parser.parse_args()
    parser.exit(2, "Not implemented: this command is a repository skeleton.\n")
