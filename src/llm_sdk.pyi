"""Type stubs for the llm_sdk package.

Exposes the public surface of class `Small_LLM_Model`, the wrapper
around a Hugging Face causal-LM used for constrained decoding. Only
the public methods are declared here; private attributes and methods
of the underlying implementation must not be relied upon by consumers
of this package.
"""

from typing import List, Any


class Small_LLM_Model:
    """Lightweight wrapper around a Hugging Face causal-LM.

    Provides the minimal interface needed to drive token-by-token
    constrained decoding: encoding text to token IDs, decoding token
    IDs back to text, and retrieving the next-token logits for a given
    sequence of input IDs.
    """

    def __init__(self) -> None: ...
    def get_logits_from_input_ids(
            self, input_ids: List[int]) -> List[float]: ...

    def get_path_to_vocab_file(self) -> str: ...
    def encode(self, text: str) -> Any: ...
    def decode(self, token_ids: List[int]) -> str: ...
