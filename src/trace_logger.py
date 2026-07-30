"""
Tracing and logging utilities for the encoding/decoding integration bonus.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class TraceLoggerMixin:
    """
    Mixin to inject token-by-token tracing capabilities into the generator.
    """
    verbose: bool
    trace: List[Dict[str, Any]]
    _current_prompt_index: int

    def _log_step(self, stage: str, **details: Any) -> None:
        """
        Record one step of the generation process for visualization.

        Args:
            stage (str): Short label identifying which part of the
            pipeline the step belongs to (e.g. ``"function_name"``,
            ``"string_param"``, ``"number_param"``, ``"boolean_param"``).
            **details (Any): Arbitrary key/value pairs describing the
            step (e.g. the chosen token, its decoded text, how many
            candidates were still active).

        Returns:
            None
        """
        entry: Dict[str, Any] = {
            "prompt_index": self._current_prompt_index,
            "stage": stage,
            **details,
        }
        self.trace.append(entry)
        if self.verbose:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in details.items())
            print(f"    🔎 [{stage}] {detail_str}")

    def export_trace(self, path: Path) -> None:
        """
        Write the full accumulated generation trace to a JSON file.

        Args:
            path (Path): Destination file for the trace log. Parent
            directories are created automatically if missing.

        Returns:
            None
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)
