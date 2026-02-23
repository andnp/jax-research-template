"""research-store: unified artifact and checkpoint storage.

Quick-start (Local Mode — zero config required)::

    from research_store import Store
    from uuid import uuid4

    store = Store(experiment_id="my_experiment")
    uri = store.put(params, name="policy_weights", execution_id=uuid4())
    params = store.get(uri)

See :class:`~research_store.store.Store` for the full API.
"""

from research_store.store import Store
from research_store.types import ArtifactKind, StoreURI
from research_store.uri import parse_uri

__all__ = ["ArtifactKind", "Store", "StoreURI", "parse_uri"]
