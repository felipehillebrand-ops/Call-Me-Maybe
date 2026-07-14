"""Vocabulary filtering for constrained decoding."""

from typing import Set
from llm_sdk import Small_LLM_Model  # type: ignore


class VocabFilter:
    """Handles pre-filtering of vocabulary tokens for constrained decoding."""

    def __init__(self, llm: Small_LLM_Model) -> None:
        """
        Initialize the filter by categorizing vocabulary tokens.

        Args:
            llm (Small_LLM_Model): The loaded LLM instance.
        """
        dummy_logits = llm.get_logits_from_input_ids([100])
        vocab_size = len(dummy_logits)

        self.numeric_tokens: Set[int] = set()
        self.quote_tokens: Set[int] = set()
        self.all_tokens: Set[int] = set(range(vocab_size))

        for token_id in range(vocab_size):
            try:
                token_str = llm.decode([token_id])
                if not token_str:
                    continue

                if all(c in "0123456789.- " for c in token_str):
                    self.numeric_tokens.add(token_id)

                if '"' in token_str:
                    self.quote_tokens.add(token_id)
            except Exception:
                continue

        self.stop_quote_ids: Set[int] = set(llm.encode('"')[0].tolist())
        self.comma_ids: Set[int] = set(llm.encode(',')[0].tolist())
        self.brace_ids: Set[int] = set(llm.encode('}')[0].tolist())
