"""Pydantic schemas for data validation and strict typing."""

from typing import Any, Dict
from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Schema for a single parameter's type definition."""
    type: str


class ReturnDef(BaseModel):
    """Schema for the return type definition."""
    type: str


class FunctionDefinition(BaseModel):
    """Schema for a function definition provided to the LLM."""
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ReturnDef


class TestPrompt(BaseModel):
    """Schema for the input test prompt."""
    prompt: str


class FunctionCallOutput(BaseModel):
    """Schema for the exact final output structure."""
    prompt: str
    name: str
    parameters: Dict[str, Any]
