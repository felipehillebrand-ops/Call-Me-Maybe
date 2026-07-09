"""Command line interface module for parsing arguments."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments containing file paths.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding."
    )

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the functions definition JSON file."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the input prompts JSON file."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path to the output results JSON file."
    )

    return parser.parse_args()
