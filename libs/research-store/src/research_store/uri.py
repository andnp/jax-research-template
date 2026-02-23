"""Parsing and generation of ``research://`` URIs."""

from __future__ import annotations

import re
from uuid import UUID

from research_store.types import ArtifactKind, StoreURI

# research://<experiment_id>/<execution_id>/<artifact_name>:<version>?kind=<kind>
_URI_PATTERN = re.compile(
    r"^research://"
    r"(?P<experiment_id>[^/]+)"
    r"/(?P<execution_id>[^/]+)"
    r"/(?P<artifact_name>[^:]+)"
    r":(?P<version>\d+)"
    r"\?kind=(?P<kind>\w+)$"
)


def parse_uri(uri: str) -> StoreURI:
    """Parse a ``research://`` URI string into a :class:`StoreURI`.

    Args:
        uri: A URI string in the canonical format produced by :func:`format_uri`.

    Returns:
        The parsed :class:`StoreURI`.

    Raises:
        ValueError: If the URI does not match the expected format.
    """
    m = _URI_PATTERN.match(uri)
    if m is None:
        raise ValueError(f"Invalid research URI: {uri!r}")
    return StoreURI(
        experiment_id=m.group("experiment_id"),
        execution_id=UUID(m.group("execution_id")),
        artifact_name=m.group("artifact_name"),
        version=int(m.group("version")),
        kind=ArtifactKind(m.group("kind")),
    )
