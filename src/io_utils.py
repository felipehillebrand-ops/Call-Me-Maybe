"""Utility functions for safe JSON file reading and writing."""

import json
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.schemas import FunctionDefinition, TestPrompt, FunctionCallOutput


def load_functions(file_path: Path) -> List[FunctionDefinition]:
    """
    Load and validate function definitions from a JSON file.

    Args:
        file_path (Path): Path to the functions JSON file.

    Returns:
        List[FunctionDefinition]: A list of validated function schemas.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or does not match
            the expected schema.
    """
    with file_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON syntax in '{file_path}': {e}"
            ) from e

    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(
            f"'{file_path}' does not match the expected function "
            f"definition schema: {e}"
        ) from e


def load_prompts(file_path: Path) -> List[TestPrompt]:
    """
    Load and validate test prompts from a JSON file.

    Args:
        file_path (Path): Path to the prompts JSON file.

    Returns:
        List[TestPrompt]: A list of validated prompt schemas.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or does not match
            the expected schema.
    """
    with file_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON syntax in '{file_path}': {e}"
            ) from e

    try:
        return [TestPrompt.model_validate(item) for item in data]
    except ValidationError as e:
        raise ValueError(
            f"'{file_path}' does not match the expected test prompt "
            f"schema: {e}"
        ) from e


def save_results(results: List[FunctionCallOutput], file_path: Path) -> None:
    """
    Save the function calling results to a JSON file safely.

    Args:
        results (List[FunctionCallOutput]): Validated results to save.
        file_path (Path): Destination path for the JSON file.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json_data = [result.model_dump() for result in results]
        json.dump(json_data, f, indent=2, ensure_ascii=False)
