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
    tree: jax.Array
    max_priority: jax.Array
