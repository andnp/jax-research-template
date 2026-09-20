"""Shared serialization helpers for experiment database persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def _json_stable(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _hash_dict(d: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_stable(d).encode()).hexdigest()
