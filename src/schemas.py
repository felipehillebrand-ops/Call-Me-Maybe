"""Pydantic schemas for data validation and strict typing."""

from typing import Any, Dict
from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Schema for a single parameter's type definition

    Attributes:
        type (str): The JSON type of the parameter (e.g. ``"string"``,
            ``"number"``, ``"integer"``, ``"boolean"``).
    """
    type: str


class ReturnDef(BaseModel):
    """Schema for the return type definition.

    Attributes:
        type (str): The JSON type returned by the function (e.g.
            ``"string"``, ``"number"``).
    """
    type: str


class FunctionDefinition(BaseModel):
    """Schema for a function definition provided to the LLM.

    Attributes:
        name (str): The function's identifier (e.g. ``"fn_add_numbers"``),
            used verbatim in the generated output.
        description (str): Natural language explanation of what the
            function does, shown to the LLM as context.
        parameters (Dict[str, ParameterDef]): Mapping of parameter name
            to its type definition.
        returns (ReturnDef): The function's declared return type.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ReturnDef


class TestPrompt(BaseModel):
    """Schema for the input test prompt.

    Attributes:
        prompt (str): The natural language request to process (e.g.
            ``"What is the sum of 2 and 3?"``).
    """
    prompt: str


class FunctionCallOutput(BaseModel):
    """Schema for the exact final output structure.

    Attributes:
        prompt (str): The original natural-language request, copied
            verbatim from the matching :class:`TestPrompt`.
        name (str): The name of the selected function, matching one of
            the ``name`` values in ``functions_definition.json``.
        parameters (Dict[str, Any]): The generated arguments for the
            selected function, keyed by parameter name, with values
            already cast to their appropriate Python type (str, int,
            float, or bool).
    """
    prompt: str
    name: str
    parameters: Dict[str, Any]
