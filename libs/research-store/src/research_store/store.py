"""Top-level ``Store`` class — the primary entry-point for research-store."""

from __future__ import annotations

import uuid
from pathlib import Path

from research_store.local_backend import LocalBackend
from research_store.types import StoreURI

_DEFAULT_ROOT = Path.home() / ".research_store"
_DEFAULT_EXPERIMENT_ID = "default"


class Store:
    """Unified artifact store with a local filesystem backend.

    In *Local Mode* (the default) all artifacts are written to a directory
    tree rooted at *root* and zero external infrastructure is required.

    Args:
        root: Filesystem root for all stored artifacts.
              Defaults to ``~/.research_store``.
        experiment_id: A human-readable label shared by a family of related
                       executions (e.g. the name of the experiment script).
                       Defaults to ``"default"``.

    Example::

        store = Store(experiment_id="ppo_cartpole")
        uri = store.put(params, name="policy_weights", execution_id=run_id)
        params_recovered = store.get(uri)
    """

    def __init__(
        self,
        root: Path = _DEFAULT_ROOT,
        experiment_id: str = _DEFAULT_EXPERIMENT_ID,
    ) -> None:
        self._backend = LocalBackend(root)
        self._experiment_id = experiment_id

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def put(self, blob: object, *, name: str, execution_id: uuid.UUID) -> StoreURI:
        """Persist *blob* and return a canonical ``research://`` URI.

        Serialization is chosen automatically:

        * Objects that are JAX pytrees with at least one ``jax.Array`` leaf
          are saved via **Orbax** (suitable for Flax / Optax model parameters).
        * All other objects are serialized with **pickle**.

        Writes are atomic: data is staged under a hidden ``.tmp_*`` path and
        only renamed into place after a successful write, preventing partially-
        written artifacts.

        Args:
            blob: The Python object to store.
            name: Short descriptive label for the artifact (e.g.
                  ``"policy_weights"``).
            execution_id: UUID identifying the current execution/run.

        Returns:
            A :class:`~research_store.types.StoreURI` pointing to the artifact.
        """
        return self._backend.put(
            blob,
            experiment_id=self._experiment_id,
            execution_id=execution_id,
            artifact_name=name,
        )

    def get(self, uri: StoreURI) -> object:
        """Retrieve and deserialize the artifact identified by *uri*.

        Args:
            uri: A :class:`~research_store.types.StoreURI` previously returned
                 by :meth:`put`.

        Returns:
            The deserialized Python object.

        Raises:
            FileNotFoundError: If no artifact exists for *uri*.
        """
        return self._backend.get(uri)

    def sync(self) -> None:
        """Flush all pending writes to the backing store.

        In *Local Mode* this is a no-op because every write is already
        synchronous and complete by the time :meth:`put` returns.  In future
        remote backends (S3/GCS) this will block until all staged uploads have
        finished, making it safe to call at the end of a Slurm job.
        """
