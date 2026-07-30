"""Parameter decoding strategies for specific data types."""

from typing import TYPE_CHECKING, Dict, List, Union

if TYPE_CHECKING:
    from src.generator import ConstrainedGenerator

STOP_BIAS_PER_TOKEN = 0.6
STOP_BIAS_GRACE_TOKENS = 1


def generate_string_parameter(
    gen: "ConstrainedGenerator",
    input_ids: List[int],
    prompt_text: str,
    p_name: str
) -> str:
    """
    Generates an unconstrained string extracted from the prompt.

    Args:
        gen (ConstrainedGenerator): The generator instance providing LLM
        and filter context.
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
    from src.generator import is_degenerate_repeat

    input_ids.extend(gen.llm.encode('"')[0].tolist())
    val_tokens: List[int] = []
    penalty_counts: Dict[int, int] = {}

    allowed_str_set = set(gen.filter.all_tokens)

    newline_ids = (
        {t for t in allowed_str_set if "\n" in gen.llm.decode([t])
         or "\\n" in gen.llm.decode([t])}
    )
    free_allowed_str = list(
        (allowed_str_set - newline_ids)
        | gen.filter.stop_quote_ids
    )

    anchor_starts = list(range(len(prompt_text)))

    for _ in range(25):
        decoded_so_far = gen.llm.decode(val_tokens)
        quote_parity_open = decoded_so_far.count('"') % 2 == 1
        extending_candidates: Dict[int, List[int]] = {}

        for t in free_allowed_str:
            piece = gen.llm.decode([t])
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
            stop_id = next(iter(gen.filter.stop_quote_ids))
            input_ids.append(stop_id)
            gen._log_step(
                "string_param",
                param=p_name,
                reason="no_extending_candidates",
                value_so_far=decoded_so_far,
            )
            break

        candidate_tokens = list(extending_candidates.keys())
        stop_token_ids_this_step = set()
        if can_stop_now:
            stop_token_ids_this_step = gen.filter.stop_quote_ids
            candidate_tokens = (
                candidate_tokens
                + list(gen.filter.stop_quote_ids)
            )

        stop_bias_value = max(
            0,
            len(val_tokens) - STOP_BIAS_GRACE_TOKENS
        ) * STOP_BIAS_PER_TOKEN

        t_id = gen._select_next_token(
            input_ids,
            candidate_tokens,
            penalty_counts,
            stop_token_ids=stop_token_ids_this_step,
            stop_bias=stop_bias_value,
        )

        if t_id in stop_token_ids_this_step:
            input_ids.append(t_id)
            gen._log_step(
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

        gen._log_step(
            "string_param",
            param=p_name,
            token_piece=gen.llm.decode([t_id]),
            candidates_remaining=len(extending_candidates),
            stop_bias=round(stop_bias_value, 2),
        )

        if is_degenerate_repeat(val_tokens):
            stop_id = next(iter(gen.filter.stop_quote_ids))
            input_ids.append(stop_id)
            gen._log_step(
                "string_param",
                param=p_name,
                reason="degenerate_repeat_detected",
                value_so_far=gen.llm.decode(val_tokens),
            )
            break

    else:
        stop_id = next(iter(gen.filter.stop_quote_ids))
        input_ids.append(stop_id)
        gen._log_step(
            "string_param",
            param=p_name,
            reason="max_steps_reached_forced_close",
            value_so_far=gen.llm.decode(val_tokens),
        )

    decoded_val = gen.llm.decode(val_tokens).strip()
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

    gen._log_step(
        "string_param", param=p_name, final_value=decoded_val
    )
    return decoded_val


def generate_number_parameter(
    gen: "ConstrainedGenerator",
    input_ids: List[int],
    p_type: str
) -> Union[int, float]:
    """
    Generates a numeric parameter resolving its float or int status.

    Args:
        gen (ConstrainedGenerator): The generator instance providing LLM
        and filter context.
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
            allowed_num = list(gen.filter.numeric_tokens)
        else:
            allowed_num = list(
                gen.filter.numeric_tokens
                | gen.filter.comma_ids
                | gen.filter.brace_ids
            )

        t_id = gen._select_next_token(input_ids, allowed_num)
        input_ids.append(t_id)

        if (t_id in gen.filter.comma_ids
                or t_id in gen.filter.brace_ids):
            gen._log_step(
                "number_param",
                step=step_idx,
                reason="delimiter_reached",
                token_piece=gen.llm.decode([t_id]),
            )
            break
        val_tokens.append(t_id)

        gen._log_step(
            "number_param",
            step=step_idx,
            token_piece=gen.llm.decode([t_id]),
        )

    num_str = gen.llm.decode(val_tokens).strip()
    num_str = "".join(c for c in num_str if c in "0123456789.-")

    if not num_str or num_str in ("-", "."):
        result: Union[int, float]
        if p_type == "number":
            result = 0.0
        else:
            result = 0
    elif p_type == "number" or "." in num_str:
        result = float(num_str)
    else:
        result = int(num_str)

    gen._log_step("number_param", final_value=result)
    return result


def generate_boolean_parameter(
    gen: "ConstrainedGenerator",
    input_ids: List[int]
) -> bool:
    """
    Forces the LLM to generate exactly one of the two valid JSON
    boolean literals ("true" or "false") using constrained decoding.

    Args:
        gen (ConstrainedGenerator): The generator instance providing LLM
        and filter context.
        input_ids (List[int]): Current sequence of token IDs. Mutated
        in place with the tokens chosen for the boolean literal.

    Returns:
        bool: The selected boolean value.
    """
    bool_sequences = {
        "true": gen.llm.encode("true")[0].tolist(),
        "false": gen.llm.encode("false")[0].tolist(),
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

        t_id = gen._select_next_token(input_ids, allowed_b)
        input_ids.append(t_id)

        active_vals = [
            v for v in active_vals
            if (step_b < len(bool_sequences[v])
                and bool_sequences[v][step_b] == t_id)
        ]
        step_b += 1

        gen._log_step(
            "boolean_param",
            step=step_b,
            token_piece=gen.llm.decode([t_id]),
            candidates_remaining=len(active_vals),
        )

        if (len(active_vals) == 1
                and step_b >= len(bool_sequences[active_vals[0]])):
            break

    if active_vals:
        chosen_val = active_vals[0]
    else:
        chosen_val = "false"
    gen._log_step("boolean_param", final_value=chosen_val)
    return chosen_val == "true"
