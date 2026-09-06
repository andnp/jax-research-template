"""Shared pydantic validation helpers."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def validate_json(model: type[_ModelT], data: str | bytes) -> _ModelT | None:
    """Validate JSON against a pydantic model, returning None instead of raising."""
    try:
        return model.model_validate_json(data)
    except ValidationError:
        return None
