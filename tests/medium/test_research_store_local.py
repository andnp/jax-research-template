"""Medium integration tests for research-store local backend.

These tests exercise actual filesystem I/O and Orbax/pickle serialization
but avoid full training runs.  Each test runs in a temporary directory so
there is no shared state.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import jax.numpy as jnp
import pytest
from research_store.store import Store
from research_store.types import ArtifactKind


@pytest.fixture()
def store(tmp_path: Path) -> Store:
    """A :class:`Store` wired to a fresh temporary directory."""
    return Store(root=tmp_path, experiment_id="test_experiment")


@pytest.fixture()
def run_id() -> uuid.UUID:
    return uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


class TestPicklePutGet:
    def test_put_returns_uri(self, store: Store, run_id: uuid.UUID) -> None:
        uri = store.put({"loss": 0.5}, name="metrics", execution_id=run_id)
        assert str(uri).startswith("research://")

    def test_put_kind_is_pickle_for_plain_dict(self, store: Store, run_id: uuid.UUID) -> None:
        uri = store.put({"loss": 0.5}, name="metrics", execution_id=run_id)
        assert uri.kind is ArtifactKind.PICKLE

    def test_get_recovers_dict(self, store: Store, run_id: uuid.UUID) -> None:
        data = {"loss": 0.5, "step": 100}
        uri = store.put(data, name="metrics", execution_id=run_id)
        recovered = store.get(uri)
        assert recovered == data

    def test_get_recovers_list(self, store: Store, run_id: uuid.UUID) -> None:
        data = [1, 2, 3, "hello"]
        uri = store.put(data, name="stuff", execution_id=run_id)
        assert store.get(uri) == data

    def test_file_exists_on_disk(self, store: Store, run_id: uuid.UUID, tmp_path: Path) -> None:
        store.put(42, name="answer", execution_id=run_id)
        pkl_path = tmp_path / "test_experiment" / str(run_id) / "answer_v1.pkl"
        assert pkl_path.exists()

    def test_no_tmp_files_left_after_put(self, store: Store, run_id: uuid.UUID, tmp_path: Path) -> None:
        store.put({"x": 1}, name="data", execution_id=run_id)
        parent = tmp_path / "test_experiment" / str(run_id)
        tmp_files = list(parent.glob(".tmp_*"))
        assert tmp_files == []

    def test_versions_increment(self, store: Store, run_id: uuid.UUID) -> None:
        uri1 = store.put("first", name="log", execution_id=run_id)
        uri2 = store.put("second", name="log", execution_id=run_id)
        assert uri1.version == 1
        assert uri2.version == 2

    def test_get_missing_uri_raises(self, store: Store, run_id: uuid.UUID) -> None:
        from research_store.types import StoreURI

        ghost = StoreURI(
            experiment_id="test_experiment",
            execution_id=run_id,
            artifact_name="ghost",
            version=99,
            kind=ArtifactKind.PICKLE,
        )
        with pytest.raises(FileNotFoundError):
            store.get(ghost)


class TestOrbaxPutGet:
    def test_put_kind_is_orbax_for_jax_array(self, store: Store, run_id: uuid.UUID) -> None:
        params = {"w": jnp.ones((3, 3)), "b": jnp.zeros(3)}
        uri = store.put(params, name="params", execution_id=run_id)
        assert uri.kind is ArtifactKind.ORBAX

    def test_get_recovers_jax_pytree_shape(self, store: Store, run_id: uuid.UUID) -> None:
        params = {"w": jnp.ones((4, 4)), "b": jnp.zeros(4)}
        uri = store.put(params, name="weights", execution_id=run_id)
        recovered = store.get(uri)
        assert isinstance(recovered, dict)
        assert recovered["w"].shape == (4, 4)  # type: ignore[union-attr]
        assert recovered["b"].shape == (4,)  # type: ignore[union-attr]

    def test_checkpoint_dir_exists(self, store: Store, run_id: uuid.UUID, tmp_path: Path) -> None:
        params = {"w": jnp.array([1.0, 2.0])}
        store.put(params, name="ckpt", execution_id=run_id)
        ckpt_dir = tmp_path / "test_experiment" / str(run_id) / "ckpt_v1"
        assert ckpt_dir.is_dir()

    def test_no_tmp_dirs_left_after_put(self, store: Store, run_id: uuid.UUID, tmp_path: Path) -> None:
        params = {"w": jnp.array([1.0])}
        store.put(params, name="net", execution_id=run_id)
        parent = tmp_path / "test_experiment" / str(run_id)
        tmp_dirs = list(parent.glob(".tmp_*"))
        assert tmp_dirs == []

    def test_orbax_versions_increment(self, store: Store, run_id: uuid.UUID) -> None:
        params = {"w": jnp.ones(2)}
        uri1 = store.put(params, name="net", execution_id=run_id)
        uri2 = store.put(params, name="net", execution_id=run_id)
        assert uri1.version == 1
        assert uri2.version == 2


class TestSyncNoop:
    def test_sync_does_not_raise(self, store: Store) -> None:
        store.sync()  # should be a no-op without error
