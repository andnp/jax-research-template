"""Small (sub-millisecond) unit tests for research-store URI parsing and generation.

These tests exercise pure logic and contain no JAX compilation or I/O.
"""

from __future__ import annotations

import uuid

import pytest
from research_store.types import ArtifactKind, StoreURI
from research_store.uri import parse_uri

_EXEC_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
_EXP_ID = "ppo_cartpole"


def _make_uri(kind: ArtifactKind = ArtifactKind.PICKLE) -> StoreURI:
    return StoreURI(
        experiment_id=_EXP_ID,
        execution_id=_EXEC_ID,
        artifact_name="policy_weights",
        version=1,
        kind=kind,
    )


class TestStoreURIFormat:
    def test_str_contains_scheme(self) -> None:
        uri = _make_uri()
        assert str(uri).startswith("research://")

    def test_str_contains_experiment_id(self) -> None:
        uri = _make_uri()
        assert _EXP_ID in str(uri)

    def test_str_contains_execution_id(self) -> None:
        uri = _make_uri()
        assert str(_EXEC_ID) in str(uri)

    def test_str_contains_artifact_name(self) -> None:
        uri = _make_uri()
        assert "policy_weights" in str(uri)

    def test_str_contains_version(self) -> None:
        uri = _make_uri()
        assert ":1" in str(uri)

    def test_str_contains_kind_pickle(self) -> None:
        uri = _make_uri(ArtifactKind.PICKLE)
        assert "kind=pickle" in str(uri)

    def test_str_contains_kind_orbax(self) -> None:
        uri = _make_uri(ArtifactKind.ORBAX)
        assert "kind=orbax" in str(uri)


class TestParseURI:
    def test_roundtrip_pickle(self) -> None:
        original = _make_uri(ArtifactKind.PICKLE)
        parsed = parse_uri(str(original))
        assert parsed == original

    def test_roundtrip_orbax(self) -> None:
        original = _make_uri(ArtifactKind.ORBAX)
        parsed = parse_uri(str(original))
        assert parsed == original

    def test_parse_experiment_id(self) -> None:
        uri = _make_uri()
        assert parse_uri(str(uri)).experiment_id == _EXP_ID

    def test_parse_execution_id(self) -> None:
        uri = _make_uri()
        assert parse_uri(str(uri)).execution_id == _EXEC_ID

    def test_parse_artifact_name(self) -> None:
        uri = _make_uri()
        assert parse_uri(str(uri)).artifact_name == "policy_weights"

    def test_parse_version(self) -> None:
        uri = _make_uri()
        assert parse_uri(str(uri)).version == 1

    def test_invalid_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid research URI"):
            parse_uri("https://not-a-research-uri")

    def test_missing_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid research URI"):
            parse_uri(f"research://{_EXP_ID}/{_EXEC_ID}/policy_weights:1")

    def test_version_is_int(self) -> None:
        uri = StoreURI(
            experiment_id=_EXP_ID,
            execution_id=_EXEC_ID,
            artifact_name="params",
            version=42,
            kind=ArtifactKind.PICKLE,
        )
        assert parse_uri(str(uri)).version == 42
