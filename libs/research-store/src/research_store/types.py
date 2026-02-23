"""Core types for research-store."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from uuid import UUID


class ArtifactKind(enum.Enum):
    """Serialization strategy for a stored artifact."""

    ORBAX = "orbax"
    """JAX pytree serialized via Orbax (saves to a directory)."""

    PICKLE = "pickle"
    """Generic Python object serialized via pickle (saves to a .pkl file)."""


@dataclass(frozen=True, slots=True)
class StoreURI:
    """A parsed ``research://`` URI identifying a single versioned artifact.

    Format::

        research://<experiment_id>/<execution_id>/<artifact_name>:<version>

    Example::

        research://exp-abc123/exec-def456/policy_weights:1
    """

    experiment_id: str
    execution_id: UUID
    artifact_name: str
    version: int
    kind: ArtifactKind

    def __str__(self) -> str:
        return f"research://{self.experiment_id}/{self.execution_id}/{self.artifact_name}:{self.version}?kind={self.kind.value}"
