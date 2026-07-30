"""Command line interface module for parsing arguments."""

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments containing
        file paths and model configuration.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool using constrained decoding."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help=(
            "HuggingFace model ID or local directory path (default: "
            "'Qwen/Qwen3-0.6B'). "
            "Allows running constrained decoding with alternative "
            "language models."
        )
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

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print a live, step-by-step view of the constrained "
            "decoding process (function/parameter narrowing, chosen "
            "tokens, stop conditions) as each prompt is processed."
        )
    )

    parser.add_argument(
        "--trace-output",
        type=Path,
        default=None,
        help=(
            "Optional path to save the full generation trace as JSON, "
            "for later inspection or visualization. Recorded regardless "
            "of --verbose; only requires --trace-output to be set."
        )
    )

    return parser.parse_args()
