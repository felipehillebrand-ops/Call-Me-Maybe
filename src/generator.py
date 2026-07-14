"""Constrained decoding generator for structured function calling."""

from typing import List, Any, Dict
import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore
from src.schemas import FunctionDefinition, TestPrompt, FunctionCallOutput
from src.vocab import VocabFilter


class ConstrainedGenerator:
    """Generator that uses token-level constraints to guide LLM inference."""

    def __init__(self, llm: Small_LLM_Model) -> None:
        """
        Initialize the generator with the LLM and pre-filtered vocabulary.

        Args:
            llm (Small_LLM_Model): The loaded language model.
        """
        self.llm = llm
        print("🔍 Pre-filtering vocabulary tokens (this may take a moment)...")
        self.filter = VocabFilter(llm)
        print("✅ Vocabulary filtering complete.")

    def _select_next_token(self, input_ids: List[int],
                           allowed_tokens: List[int]) -> int:
        """
        Get logits from the LLM, apply mask, and select the highest probability
        token (Optimized with NumPy vectorization for extreme speed).

        Args:
            input_ids (List[int]): Current sequence of token IDs.
            allowed_tokens (List[int]): List of allowed token IDs for this
            step.

        Returns:
            int: The selected token ID.
        """
        logits = self.llm.get_logits_from_input_ids(input_ids)
        logits_arr = np.array(logits)
        masked_logits = np.full(len(logits), -np.inf)

        allowed_arr = np.array(allowed_tokens)
        valid_indices = allowed_arr[allowed_arr < len(logits)]

        masked_logits[valid_indices] = logits_arr[valid_indices]

        return int(np.argmax(masked_logits))

    def generate_call(
            self, test_prompt: TestPrompt,
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
        prompt_text = test_prompt.prompt

        context = (
            "You are a function calling assistant. Choose the correct "
            "function and extract its parameters from the user prompt."
            "\n\nAvailable functions:\n"
        )
        for fn in functions:
            params = ", ".join(
                f"{k}: {v.type}" for k, v in fn.parameters.items()
            )
            context += f"- {fn.name}({params}): {fn.description}\n"
        context += f"\nUser prompt: {prompt_text}\nOutput JSON:\n"
        context += f'{{\n  "prompt": "{prompt_text}",\n  "name": "'

        input_ids = self.llm.encode(context)[0].tolist()

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
                n
                for n in active_names
                if step < len(name_sequences[n])
                and name_sequences[n][step] == token_id
            ]
            step += 1

            if (len(active_names) == 1
                    and step >= len(name_sequences[active_names[0]])):
                chosen_name = active_names[0]
                break

        if not chosen_name and active_names:
            chosen_name = active_names[0]
        if not chosen_name:
            chosen_name = functions[0].name

        matched_fn = next(fn for fn in functions if fn.name == chosen_name)
        input_ids += self.llm.encode('",\n  "parameters": {\n')[0].tolist()

        generated_params: Dict[str, Any] = {}
        param_items = list(matched_fn.parameters.items())

        for idx, (p_name, p_def) in enumerate(param_items):
            input_ids += self.llm.encode(f'    "{p_name}": ')[0].tolist()

            if p_def.type == "string":
                input_ids += self.llm.encode('"')[0].tolist()
                val_tokens: List[int] = []

                allowed_str_set = (
                    set(self.filter.all_tokens) - self.filter.quote_tokens
                )
                newline_ids = (
                    {t for t in allowed_str_set if "\n" in self.llm.decode([t])
                     or "\\n" in self.llm.decode([t])}
                )
                allowed_str = list(
                    (allowed_str_set - newline_ids)
                    | self.filter.stop_quote_ids
                )

                for _ in range(150):
                    t_id = self._select_next_token(input_ids, allowed_str)

                    if t_id in self.filter.stop_quote_ids:
                        input_ids.append(t_id)
                        break

                    input_ids.append(t_id)
                    val_tokens.append(t_id)

                generated_params[p_name] = (
                    self.llm.decode(val_tokens).strip().replace("\\\\", "\\")
                )

            elif p_def.type in ("number", "integer"):
                val_tokens = []

                for step_idx in range(25):
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
                        break
                    val_tokens.append(t_id)

                num_str = self.llm.decode(val_tokens).strip()
                num_str = "".join(c for c in num_str if c in "0123456789.-")

                if not num_str or num_str in ("-", "."):
                    generated_params[p_name] = (
                        0.0 if p_def.type == "number" else 0
                    )
                else:
                    if p_def.type == "number" or "." in num_str:
                        generated_params[p_name] = float(num_str)
                    else:
                        generated_params[p_name] = int(num_str)

            if idx < len(param_items) - 1:
                input_ids += self.llm.encode(",\n")[0].tolist()
            else:
                input_ids += self.llm.encode("\n")[0].tolist()

        return FunctionCallOutput(
            prompt=prompt_text,
            name=chosen_name,
            parameters=generated_params
        )
