"""Constrained decoding generator for structured function calling."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from llm_sdk import Small_LLM_Model
from src.schemas import FunctionDefinition, TestPrompt, FunctionCallOutput
from src.vocab import VocabFilter


REPETITION_PENALTY = 1.8
MAX_NGRAM_REPEATS = 2
NGRAM_SIZE = 3
STOP_BIAS_PER_TOKEN = 0.6
STOP_BIAS_GRACE_TOKENS = 1


class ConstrainedGenerator:
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

    @staticmethod
    def _is_degenerate_repeat(val_tokens: List[int]) -> bool:
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

    def _generate_string_parameter(
        self,
        input_ids: List[int],
        prompt_text: str,
        p_name: str
    ) -> str:
        """
        Generates an unconstrained string extracted from the prompt.

        Args:
            input_ids (List[int]): Current sequence of token IDs. Mutated
            in place with the opening/closing quotes and the tokens
            chosen for the string value.
            prompt_text (str): The original natural-language prompt used
            as the source text the generated value must be anchored to.
            p_name (str): The parameter's name, used to strip trailing
            words in the generated value that just echo the parameter
            name.

        Returns:
            str: The extracted and cleaned string value.
        """
        input_ids.extend(self.llm.encode('"')[0].tolist())
        val_tokens: List[int] = []
        penalty_counts: Dict[int, int] = {}

        allowed_str_set = set(self.filter.all_tokens)

        newline_ids = (
            {t for t in allowed_str_set if "\n" in self.llm.decode([t])
             or "\\n" in self.llm.decode([t])}
        )
        free_allowed_str = list(
            (allowed_str_set - newline_ids)
            | self.filter.stop_quote_ids
        )

        anchor_starts = list(range(len(prompt_text)))

        for _ in range(25):
            decoded_so_far = self.llm.decode(val_tokens)
            quote_parity_open = decoded_so_far.count('"') % 2 == 1
            extending_candidates: Dict[int, List[int]] = {}

            for t in free_allowed_str:
                piece = self.llm.decode([t])
                if not piece:
                    continue
                candidate_text = decoded_so_far + piece
                valid_starts = [
                    s for s in anchor_starts
                    if prompt_text.startswith(candidate_text, s)
                ]
                if valid_starts:
                    extending_candidates[t] = valid_starts

            can_stop_now = bool(decoded_so_far) and not quote_parity_open

            if not extending_candidates:
                stop_id = next(iter(self.filter.stop_quote_ids))
                input_ids.append(stop_id)
                self._log_step(
                    "string_param",
                    param=p_name,
                    reason="no_extending_candidates",
                    value_so_far=decoded_so_far,
                )
                break

            candidate_tokens = list(extending_candidates.keys())
            stop_token_ids_this_step = set()
            if can_stop_now:
                stop_token_ids_this_step = self.filter.stop_quote_ids
                candidate_tokens = (
                    candidate_tokens
                    + list(self.filter.stop_quote_ids)
                )

            stop_bias_value = max(
                0,
                len(val_tokens) - STOP_BIAS_GRACE_TOKENS
            ) * STOP_BIAS_PER_TOKEN

            t_id = self._select_next_token(
                input_ids,
                candidate_tokens,
                penalty_counts,
                stop_token_ids=stop_token_ids_this_step,
                stop_bias=stop_bias_value,
            )

            if t_id in stop_token_ids_this_step:
                input_ids.append(t_id)
                self._log_step(
                    "string_param",
                    param=p_name,
                    reason="stop_token_selected",
                    value_so_far=decoded_so_far,
                )
                break

            input_ids.append(t_id)
            val_tokens.append(t_id)
            penalty_counts[t_id] = penalty_counts.get(t_id, 0) + 1
            anchor_starts = extending_candidates[t_id]

            self._log_step(
                "string_param",
                param=p_name,
                token_piece=self.llm.decode([t_id]),
                candidates_remaining=len(extending_candidates),
                stop_bias=round(stop_bias_value, 2),
            )

            if self._is_degenerate_repeat(val_tokens):
                stop_id = next(iter(self.filter.stop_quote_ids))
                input_ids.append(stop_id)
                self._log_step(
                    "string_param",
                    param=p_name,
                    reason="degenerate_repeat_detected",
                    value_so_far=self.llm.decode(val_tokens),
                )
                break

        decoded_val = self.llm.decode(val_tokens).strip()
        decoded_val = (
            decoded_val
            .replace('\\"', '"')
            .replace('\\\\', '\\')
            .replace('\\n', '')
            .strip()
        )
        param_name_words = p_name.lower().replace("_", " ").split()
        val_words = decoded_val.split()
        while (
            val_words
            and val_words[-1].lower() in param_name_words
            and len(val_words) > 1
        ):
            val_words = val_words[:-1]
        decoded_val = " ".join(val_words)

        self._log_step(
            "string_param", param=p_name, final_value=decoded_val
        )
        return decoded_val

    def _generate_number_parameter(
        self,
        input_ids: List[int],
        p_type: str
    ) -> Union[int, float]:
        """
        Generates a numeric parameter resolving its float or int status.

        Args:
            input_ids (List[int]): Current sequence of token IDs. Mutated
            in place with the tokens chosen for the numeric value and the
            delimiter that closed it.
            p_type (str): The declared parameter type, either ``"number"``
            (always returned as float) or ``"integer"``.

        Returns:
            Union[int, float]: The generated numeric value, or ``0``/
            ``0.0`` if no usable digits were produced.
        """
        val_tokens = []

        for step_idx in range(15):
            if step_idx == 0:
                allowed_num = list(self.filter.numeric_tokens)
            else:
                allowed_num = list(
                    self.filter.numeric_tokens
                    | self.filter.comma_ids
                    | self.filter.brace_ids
                )

            t_id = self._select_next_token(input_ids, allowed_num)
            input_ids.append(t_id)

            if (t_id in self.filter.comma_ids
                    or t_id in self.filter.brace_ids):
                self._log_step(
                    "number_param",
                    step=step_idx,
                    reason="delimiter_reached",
                    token_piece=self.llm.decode([t_id]),
                )
                break
            val_tokens.append(t_id)

            self._log_step(
                "number_param",
                step=step_idx,
                token_piece=self.llm.decode([t_id]),
            )

        num_str = self.llm.decode(val_tokens).strip()
        num_str = "".join(c for c in num_str if c in "0123456789.-")

        if not num_str or num_str in ("-", "."):
            result: Union[int, float] = 0.0 if p_type == "number" else 0
        elif p_type == "number" or "." in num_str:
            result = float(num_str)
        else:
            result = int(num_str)

        self._log_step("number_param", final_value=result)
        return result

    def _generate_boolean_parameter(
        self,
        input_ids: List[int]
    ) -> bool:
        """
        Forces the LLM to generate exactly one of the two valid JSON
        boolean literals ("true" or "false") using constrained decoding.

        Args:
            input_ids (List[int]): Current sequence of token IDs. Mutated
            in place with the tokens chosen for the boolean literal.

        Returns:
            bool: The selected boolean value.
        """
        bool_sequences = {
            "true": self.llm.encode("true")[0].tolist(),
            "false": self.llm.encode("false")[0].tolist(),
        }
        active_vals = list(bool_sequences.keys())
        step_b = 0

        while active_vals:
            allowed_b = list({
                bool_sequences[v][step_b]
                for v in active_vals
                if step_b < len(bool_sequences[v])
            })
            if not allowed_b:
                break

            t_id = self._select_next_token(input_ids, allowed_b)
            input_ids.append(t_id)

            active_vals = [
                v for v in active_vals
                if (step_b < len(bool_sequences[v])
                    and bool_sequences[v][step_b] == t_id)
            ]
            step_b += 1

            self._log_step(
                "boolean_param",
                step=step_b,
                token_piece=self.llm.decode([t_id]),
                candidates_remaining=len(active_vals),
            )

            if (len(active_vals) == 1
                    and step_b >= len(bool_sequences[active_vals[0]])):
                break

        chosen_val = active_vals[0] if active_vals else "false"
        self._log_step("boolean_param", final_value=chosen_val)
        return chosen_val == "true"

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
            "information ends.\n\n"
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
                generated_params[p_name] = self._generate_string_parameter(
                    input_ids, prompt_text, p_name
                )

            elif p_def.type in ("number", "integer"):
                generated_params[p_name] = self._generate_number_parameter(
                    input_ids, p_def.type
                )

            elif p_def.type == "boolean":
                generated_params[p_name] = self._generate_boolean_parameter(
                    input_ids
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
