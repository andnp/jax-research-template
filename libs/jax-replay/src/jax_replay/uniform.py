import jax
import jax.numpy as jnp

from jax_replay.types import BufferState


def init_buffer(prototype: object, capacity: int) -> BufferState:
    leaves, _ = jax.tree.flatten(prototype)

    flat_data = {}
    for i, leaf in enumerate(leaves):
        flat_data[str(i)] = jnp.zeros((capacity, *leaf.shape), dtype=leaf.dtype)

    return BufferState(
        data=flat_data,
        pointer=jnp.array(0, dtype=jnp.uint32),
        count=jnp.array(0, dtype=jnp.uint32),
    )


def add(state: BufferState, transition: object) -> BufferState:
    leaves, _ = jax.tree.flatten(transition)
    capacity = jax.tree.leaves(state.data)[0].shape[0]
    idx = state.pointer % jnp.uint32(capacity)

    new_data = {}
    for i, leaf in enumerate(leaves):
        key = str(i)
        new_data[key] = state.data[key].at[idx].set(leaf)

    return state._replace(
        data=new_data,
        pointer=state.pointer + jnp.uint32(1),
        count=jnp.minimum(state.count + jnp.uint32(1), jnp.uint32(capacity)),
    )


def sample[T](state: BufferState, key: jax.Array, batch_size: int, prototype: T) -> T:
    indices = jax.random.randint(key, (batch_size,), 0, state.count.astype(jnp.int32))
    _, treedef = jax.tree.flatten(prototype)

    sampled_leaves = []
    for i in range(len(treedef.flatten_up_to(prototype))):
        sampled_leaves.append(state.data[str(i)][indices])

    return treedef.unflatten(sampled_leaves)
