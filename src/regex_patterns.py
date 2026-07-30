"""Regex patterns and mapping for constrained decoding."""

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from src.parameter_decoders import generate_string_parameter

if TYPE_CHECKING:
    from src.generator import ConstrainedGenerator

REGEX_KEYWORD_MAP: Dict[str, str] = {
    "vowels": "[aeiouAEIOU]",
    "vowel": "[aeiouAEIOU]",
    "consonants": "[^aeiouAEIOU\\s]",
    "consonant": "[^aeiouAEIOU\\s]",
    "numbers": r"\d+",
    "number": r"\d+",
    "digits": r"\d+",
    "digit": r"\d",
    "integers": r"\d+",
    "integer": r"\d+",
    "whitespace": r"\s+",
    "spaces": r"\s+",
    "space": r"\s+",
    "blank spaces": r"\s+",
    "tabs": r"\t+",
    "newlines": r"\n+",
    "line breaks": r"\n+",
    "punctuation": r"[.,!?;:]",
    "hyphens": r"-+",
    "hyphen": r"-+",
    "dashes": r"-+",
    "dash": r"-+",
    "underscores": r"_+",
    "quotes": r"[\"']",
    "quotation marks": r"[\"']",
    "parentheses": r"[()]",
    "brackets": r"[\[\]]",
    "braces": r"[{}]",
    "special characters": r"[^a-zA-Z0-9\s]",
    "special symbols": r"[^a-zA-Z0-9\s]",
    "symbols": r"[^a-zA-Z0-9\s]",
    "non-alphanumeric characters": r"[^a-zA-Z0-9]",
    "letters": "[a-zA-Z]",
    "alphabetic characters": "[a-zA-Z]",
    "uppercase letters": "[A-Z]",
    "uppercase": "[A-Z]",
    "capital letters": "[A-Z]",
    "lowercase letters": "[a-z]",
    "lowercase": "[a-z]",
    "words": r"\w+",
    "word characters": r"\w+",
    "alphanumeric characters": r"[a-zA-Z0-9]+",
    "alphanumeric": r"[a-zA-Z0-9]+",
    "non-word characters": r"\W+",
}


def detect_known_regex_keyword(
    prompt_text: str
) -> Optional[Tuple[str, str]]:
    """
    Look for a whole-word match of a known character-class keyword
    (e.g. "vowels", "digits") in the prompt text.

    Args:
        prompt_text (str): The original natural-language prompt.

    Returns:
        Optional[Tuple[str, str]]: A ``(keyword, regex_pattern)`` pair
        for the longest matching keyword, or ``None`` if no known
        keyword appears in the prompt. Longer keywords are checked
        first so e.g. "vowels" wins over a hypothetical shorter
        overlapping entry.
    """
    lowered = prompt_text.lower()
    for keyword in sorted(REGEX_KEYWORD_MAP, key=len, reverse=True):
        if re.search(rf"\b{re.escape(keyword)}\b", lowered):
            return keyword, REGEX_KEYWORD_MAP[keyword]
    return None


def generate_regex_parameter(
    gen: "ConstrainedGenerator",
    input_ids: List[int],
    prompt_text: str,
    p_name: str
) -> str:
    """
    Generates a value for a parameter that is expected to hold a
    regex pattern (name containing "regex" or "pattern").

    Args:
        gen (ConstrainedGenerator): The generator instance providing LLM
        and filter context.
        input_ids (List[int]): Current sequence of token IDs. Mutated
        in place with the tokens for the opening/closing quotes and
        the generated or injected pattern.
        prompt_text (str): The original natural-language prompt.
        p_name (str): The parameter's name, used for tracing and
        passed through to the extractive fallback.

    Returns:
        str: The resulting regex pattern.
    """
    detected = detect_known_regex_keyword(prompt_text)
    if detected is not None:
        keyword, pattern = detected
        input_ids.extend(gen.llm.encode(f'"{pattern}"')[0].tolist())
        gen._log_step(
            "string_param",
            param=p_name,
            reason="regex_keyword_matched",
            keyword=keyword,
            final_value=pattern,
        )
        return pattern

    return generate_string_parameter(gen, input_ids, prompt_text, p_name)
