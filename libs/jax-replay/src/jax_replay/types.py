from typing import NamedTuple

import jax


class BufferState(NamedTuple):
    data: dict[str, jax.Array]
    pointer: jax.Array
    count: jax.Array


class PERBufferState(NamedTuple):
    data: dict[str, jax.Array]
    pointer: jax.Array
    count: jax.Array
    logical_capacity: jax.Array
    storage_capacity: jax.Array
    tree: jax.Array
    max_priority: jax.Array
