"""Component model — a logical unit of code tracked by path and hash."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel


class ComponentType(StrEnum):
    ALGO = "ALGO"
    ENV = "ENV"
    WRAPPER = "WRAPPER"
    OTHER = "OTHER"


class Component(BaseModel):
    """A named code component whose source is tracked for versioning.

    Args:
        name: Human-readable identifier (e.g. "PPO").
        path: Relative or absolute path to the source file.
        type: Logical role of the component.
    """

    model_config = {"frozen": True}

    name: str
    path: Path
    type: ComponentType = ComponentType.OTHER

    @classmethod
    def from_module(
        cls, name: str, module: ModuleType, type: ComponentType = ComponentType.OTHER
    ) -> Component:
        """Build a Component whose path is resolved from an imported module's source file."""
        file = getattr(module, "__file__", None)
        if file is None:
            raise ValueError(f"module {module.__name__!r} has no source file")
        return cls(name=name, path=Path(file), type=type)

    def code_hash(self) -> str:
        """Return the SHA-256 hex digest of the source file.

        Returns an empty string when the file does not exist (e.g. in tests).
        """
        if not self.path.exists():
            return ""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()
