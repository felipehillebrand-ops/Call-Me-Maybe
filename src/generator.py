"""Constrained decoding generator for structured function calling."""

import os
from typing import Any, Dict, List, Tuple
import numpy as np

from llm_sdk import Small_LLM_Model
from src.schemas import FunctionDefinition, TestPrompt, FunctionCallOutput
from src.vocab import VocabFilter
from src.regex_patterns import generate_regex_parameter
from src.parameter_decoders import (
    generate_string_parameter,
    generate_number_parameter,
    generate_boolean_parameter,
)
from src.trace_logger import TraceLoggerMixin

REPETITION_PENALTY = 1.8
MAX_NGRAM_REPEATS = 2
NGRAM_SIZE = 3


def is_degenerate_repeat(val_tokens: List[int]) -> bool:
    """
    Detect whether the tail of the generated token sequence is looping
    on the same short n-gram, which indicates greedy-decoding
    degeneration rather than a meaningful value.

    Args:
        val_tokens (List[int]): Tokens generated so far for the current
        string value.

    Returns:
        bool: True if a repeating n-gram pattern was detected.
    """
    needed = NGRAM_SIZE * (MAX_NGRAM_REPEATS + 1)
    if len(val_tokens) < needed:
        return False

    tail = val_tokens[-needed:]
    ngram = tail[-NGRAM_SIZE:]
    for i in range(1, MAX_NGRAM_REPEATS + 1):
        start = len(tail) - NGRAM_SIZE * (i + 1)
        end = start + NGRAM_SIZE
        if tail[start:end] != ngram:
            return False
    return True


class ConstrainedGenerator(TraceLoggerMixin):
    """Generator that uses token-level constraints to guide LLM inference."""

    def __init__(
        self, llm: Small_LLM_Model, verbose: bool = False
    ) -> None:
        """
        Initialize the generator with the LLM and pre-filtered vocabulary.

        Args:
            llm (Small_LLM_Model): The loaded language model.
            verbose (bool): When True, every constrained-decoding step is
            printed to stdout as it happens, offering a live view of how
            the generation process narrows token choices down to a
            schema-valid result. Has no effect on the generated output,
            only on what is printed; the lightweight trace log is always
            recorded regardless of this flag (see :meth:`export_trace`).
        """
        self.llm = llm
        self.verbose = verbose
        self.trace: List[Dict[str, Any]] = []
        self._current_prompt_index = 0
        print("🔍 Pre-filtering vocabulary tokens (this may take a moment)...")
        self.filter = VocabFilter(llm)
        print("✅ Vocabulary filtering complete.")

    def _select_next_token(
        self,
        input_ids: List[int],
        allowed_tokens: List[int],
        penalty_counts: Dict[int, int] | None = None,
        stop_token_ids: set[int] | None = None,
        stop_bias: float = 0.0
    ) -> int:
        """
        Get logits from the LLM, apply mask (and optional repetition
        penalty / stop bias), and select the highest probability token.

        Args:
            input_ids (List[int]): Current sequence of token IDs.
            allowed_tokens (List[int]): List of allowed token IDs for this
            step.
            penalty_counts (Dict[int, int] | None): Mapping of token id ->
            number of times it has already been generated in the current
            value. When provided, repeated tokens get their logits reduced
            so the model is nudged away from repetition loops.
            stop_token_ids (set[int] | None): Token IDs that would close the
            current value. When provided together with stop_bias, their
            logits are boosted so the model is encouraged to stop once a
            reasonable value has been produced.
            stop_bias (float): Amount added to the logits of stop_token_ids.

        Returns:
            int: The selected token ID.
        """
        logits = self.llm.get_logits_from_input_ids(input_ids)
        logits_arr = np.array(logits, dtype=np.float64)
        masked_logits = np.full(len(logits), -np.inf)

        allowed_arr = np.array(allowed_tokens)
        valid_indices = allowed_arr[allowed_arr < len(logits)]

        masked_logits[valid_indices] = logits_arr[valid_indices]

        if penalty_counts:
            for tok_id, count in penalty_counts.items():
                if 0 <= tok_id < len(masked_logits) and np.isfinite(
                        masked_logits[tok_id]):
                    masked_logits[tok_id] -= REPETITION_PENALTY * count

        if stop_token_ids and stop_bias:
            for tok_id in stop_token_ids:
                if 0 <= tok_id < len(masked_logits) and np.isfinite(
                        masked_logits[tok_id]):
                    masked_logits[tok_id] += stop_bias

        return int(np.argmax(masked_logits))

    def _generate_function_name(
        self,
        input_ids: List[int],
        functions: List[FunctionDefinition]
    ) -> Tuple[str, FunctionDefinition]:
        """
        Generates the function name token by token and returns the
        matched function definition.

        Args:
            input_ids (List[int]): Current sequence of token IDs. Mutated
            in place with the tokens chosen for the function name.
            functions (List[FunctionDefinition]): Available function
            definitions to choose the name from.

        Returns:
            Tuple[str, FunctionDefinition]: The chosen function name and
            its corresponding definition.
        """
        name_sequences = {
            fn.name: self.llm.encode(fn.name)[0].tolist()
            for fn in functions
        }
        chosen_name = ""
        step = 0
        active_names = list(name_sequences.keys())

        while len(active_names) > 1 or (
                active_names
                and step < len(name_sequences[active_names[0]])):
            allowed = list({
                name_sequences[n][step]
                for n in active_names
                if step < len(name_sequences[n])
            })
            if not allowed:
                break

            token_id = self._select_next_token(input_ids, allowed)
            input_ids.append(token_id)

            active_names = [
                n for n in active_names
                if step < len(name_sequences[n])
                and name_sequences[n][step] == token_id
            ]
            step += 1

            self._log_step(
                "function_name",
                step=step,
                token_piece=self.llm.decode([token_id]),
                candidates_remaining=len(active_names),
            )

            if (len(active_names) == 1
                    and step >= len(name_sequences[active_names[0]])):
                chosen_name = active_names[0]
                break

        if not chosen_name and active_names:
            chosen_name = active_names[0]
        if not chosen_name:
            chosen_name = functions[0].name

        matched_fn = next(fn for fn in functions if fn.name == chosen_name)
        self._log_step("function_name", chosen_name=chosen_name)
        return chosen_name, matched_fn

    def generate_call(
        self,
        test_prompt: TestPrompt,
        functions: List[FunctionDefinition]
    ) -> FunctionCallOutput:
        """
        Generate a constrained function call for a given prompt.

        Args:
            test_prompt (TestPrompt): The input prompt schema.
            functions (List[FunctionDefinition]): Available functions
            definitions.

        Returns:
            FunctionCallOutput: The structured function call output.
        """
        self._current_prompt_index += 1
        prompt_text = test_prompt.prompt

        demo_indexes_raw = os.environ.get("CALL_ME_MAYBE_DEMO_FAIL_INDEXES")
        if demo_indexes_raw:
            demo_indexes = {
                int(i) for i in demo_indexes_raw.split(",")
                if i.strip().isdigit()
            }
            if self._current_prompt_index in demo_indexes:
                raise RuntimeError(
                    "Simulated failure (CALL_ME_MAYBE_DEMO_FAIL_INDEXES "
                    f"includes prompt #{self._current_prompt_index}) for "
                    "error-recovery demonstration."
                )

        if self.verbose:
            print(f"\n🧩 Prompt #{self._current_prompt_index}: "
                  f"{prompt_text!r}")
        self._log_step("prompt", prompt=prompt_text)

        escaped_prompt = prompt_text.replace('"', '\\"')

        context = (
            "<|im_start|>system\n"
            "You are a precise data extraction AI. Extract the exact "
            "parameters from the user prompt based on the function "
            "definitions. Output strictly valid JSON with no extra text. "
            "CRITICAL RULES:\n"
            "1. Do not add any conversational filler, explanations, or "
            "trailing continuations.\n"
            "2. End the parameter string exactly where the relevant "
            "information ends.\n"
            "3. If a parameter asks for a regex pattern (e.g., for "
            "'vowels' or 'numbers'), generate standard regex syntax "
            "(like '[aeiou]' or '\\d+').\n\n"
            "Available functions:\n"
        )
        for fn in functions:
            params = ", ".join(
                f"{k}: {v.type}" for k, v in fn.parameters.items()
            )
            context += f"- {fn.name}({params}): {fn.description}\n"

        context += (
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt_text}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        context += f'{{\n  "prompt": "{escaped_prompt}",\n  "name": "'

        input_ids = self.llm.encode(context)[0].tolist()

        chosen_name, matched_fn = self._generate_function_name(
            input_ids, functions
        )

        input_ids.extend(
            self.llm.encode('",\n  "parameters": {\n')[0].tolist()
        )

        generated_params: Dict[str, Any] = {}
        param_items = list(matched_fn.parameters.items())

        for idx, (p_name, p_def) in enumerate(param_items):
            input_ids.extend(self.llm.encode(f'    "{p_name}": ')[0].tolist())

            if p_def.type == "string":
                if any(kw in p_name.lower() for kw in ("regex", "pattern")):
                    generated_params[p_name] = generate_regex_parameter(
                        self, input_ids, prompt_text, p_name
                    )
                else:
                    generated_params[p_name] = generate_string_parameter(
                        self, input_ids, prompt_text, p_name
                    )

            elif p_def.type in ("number", "integer"):
                generated_params[p_name] = generate_number_parameter(
                    self, input_ids, p_def.type
                )

            elif p_def.type == "boolean":
                generated_params[p_name] = generate_boolean_parameter(
                    self, input_ids
                )

            if idx < len(param_items) - 1:
                input_ids.extend(self.llm.encode(",\n")[0].tolist())
            else:
                input_ids.extend(self.llm.encode("\n")[0].tolist())

        self._log_step(
            "result",
            name=chosen_name,
            parameters=generated_params,
        )
        if self.verbose:
            print(f"  ✅ -> {chosen_name}({generated_params})")

        return FunctionCallOutput(
            prompt=prompt_text,
            name=chosen_name,
            parameters=generated_params,
        )
