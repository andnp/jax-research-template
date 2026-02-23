"""Local filesystem backend for research-store.

Provides atomic put/get operations backed by a plain directory tree.
All writes use the "hidden staging" pattern: data is written to a
temporary path and only made visible via an atomic ``os.rename`` once
the write has completed successfully.

Directory layout::

    <root>/
      <experiment_id>/
        <execution_id>/
          <artifact_name>_v<version>/          # orbax checkpoint dir
          <artifact_name>_v<version>.pkl       # pickle artifact
"""

from __future__ import annotations

import os
import pickle
import uuid
from pathlib import Path

import orbax.checkpoint as ocp

from research_store.types import ArtifactKind, StoreURI


def _has_jax_arrays(obj: object) -> bool:
    """Return True if *obj* is a JAX pytree containing at least one ``jax.Array``."""
    try:
        import jax
    except ImportError:
        return False
    leaves = jax.tree_util.tree_leaves(obj)
    return any(isinstance(leaf, jax.Array) for leaf in leaves)


def _artifact_dir(root: Path, uri: StoreURI) -> Path:
    """Return the parent directory for the artifact described by *uri*."""
    return root / uri.experiment_id / str(uri.execution_id)


def _artifact_path(root: Path, uri: StoreURI) -> Path:
    """Return the filesystem path (file or dir) for *uri*."""
    parent = _artifact_dir(root, uri)
    stem = f"{uri.artifact_name}_v{uri.version}"
    if uri.kind is ArtifactKind.ORBAX:
        return parent / stem
    return parent / f"{stem}.pkl"


class LocalBackend:
    """Atomic filesystem backend for artifact storage.

    Args:
        root: Base directory under which all artifacts are stored.
              Created automatically on first use.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def put(self, blob: object, experiment_id: str, execution_id: uuid.UUID, artifact_name: str) -> StoreURI:
        """Atomically persist *blob* and return the canonical :class:`StoreURI`.

        Serialization strategy:
        - JAX pytrees (any pytree with ``jax.Array`` leaves) → Orbax checkpoint directory.
        - Everything else → pickle file.

        Args:
            blob: The Python object to store.
            experiment_id: Human-readable experiment identifier.
            execution_id: UUID for the specific execution/run.
            artifact_name: Short descriptive name for the artifact.

        Returns:
            A :class:`StoreURI` pointing to the stored artifact.
        """
        kind = ArtifactKind.ORBAX if _has_jax_arrays(blob) else ArtifactKind.PICKLE
        version = self._next_version(experiment_id, execution_id, artifact_name, kind)
        uri = StoreURI(
            experiment_id=experiment_id,
            execution_id=execution_id,
            artifact_name=artifact_name,
            version=version,
            kind=kind,
        )
        parent = _artifact_dir(self._root, uri)
        parent.mkdir(parents=True, exist_ok=True)

        if kind is ArtifactKind.ORBAX:
            self._put_orbax(blob, parent, uri)
        else:
            self._put_pickle(blob, parent, uri)

        return uri

    def get(self, uri: StoreURI) -> object:
        """Retrieve and deserialize the artifact identified by *uri*.

        Args:
            uri: A :class:`StoreURI` previously returned by :meth:`put`.

        Returns:
            The deserialized Python object.

        Raises:
            FileNotFoundError: If no artifact exists at the resolved path.
        """
        path = _artifact_path(self._root, uri)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}")

        if uri.kind is ArtifactKind.ORBAX:
            return self._get_orbax(path)
        return self._get_pickle(path)

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------

    def _next_version(self, experiment_id: str, execution_id: uuid.UUID, artifact_name: str, kind: ArtifactKind) -> int:
        """Return the next available version number (1-based) for an artifact."""
        parent = self._root / experiment_id / str(execution_id)
        if not parent.exists():
            return 1
        stem_prefix = f"{artifact_name}_v"
        suffix = "" if kind is ArtifactKind.ORBAX else ".pkl"
        existing: list[int] = []
        for entry in parent.iterdir():
            name = entry.name
            if name.startswith(stem_prefix):
                tail = name[len(stem_prefix):]
                if suffix:
                    tail = tail.removesuffix(suffix)
                if tail.isdigit():
                    existing.append(int(tail))
        return max(existing, default=0) + 1

    # ------------------------------------------------------------------
    # Serialization: Orbax
    # ------------------------------------------------------------------

    @staticmethod
    def _put_orbax(blob: object, parent: Path, uri: StoreURI) -> None:
        """Write a JAX pytree atomically via Orbax.

        Orbax is invoked on a temporary directory which is renamed to the
        final path only after a successful save, guaranteeing atomicity.
        """
        final_path = parent / f"{uri.artifact_name}_v{uri.version}"
        tmp_path = parent / f".tmp_{uri.artifact_name}_{uuid.uuid4().hex}"
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(tmp_path, blob)
        checkpointer.wait_until_finished()
        os.rename(tmp_path, final_path)

    @staticmethod
    def _get_orbax(path: Path) -> object:
        """Restore a JAX pytree from an Orbax checkpoint directory."""
        checkpointer = ocp.StandardCheckpointer()
        return checkpointer.restore(path)

    # ------------------------------------------------------------------
    # Serialization: Pickle
    # ------------------------------------------------------------------

    @staticmethod
    def _put_pickle(blob: object, parent: Path, uri: StoreURI) -> None:
        """Write an arbitrary Python object atomically via pickle."""
        final_path = parent / f"{uri.artifact_name}_v{uri.version}.pkl"
        tmp_path = parent / f".tmp_{uri.artifact_name}_{uuid.uuid4().hex}.pkl"
        try:
            with tmp_path.open("wb") as fh:
                pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _get_pickle(path: Path) -> object:
        """Deserialize a pickle artifact."""
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301


