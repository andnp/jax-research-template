import jax
import jax.numpy as jnp

from jax_replay.sum_tree import tree_init, tree_sample_batch, tree_update
from jax_replay.types import PERBufferState


def _next_power_of_two(capacity: int) -> int:
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    return 1 << (capacity - 1).bit_length()


def _minimum_sample_probability(state: PERBufferState, total_priority: jax.Array):
    storage_capacity = state.tree.shape[0] // 2
    all_leaf_priorities = state.tree[storage_capacity:]
    logical_capacity = state.logical_capacity.astype(jnp.uint32)
    populated_mask = jnp.arange(storage_capacity, dtype=jnp.uint32) < state.count
    valid_mask = jnp.arange(storage_capacity, dtype=jnp.uint32) < logical_capacity
    populated_mask = jnp.logical_and(populated_mask, valid_mask)
    positive_mask = jnp.logical_and(populated_mask, all_leaf_priorities > 0.0)
    masked_priorities = jnp.where(positive_mask, all_leaf_priorities, jnp.full_like(all_leaf_priorities, jnp.inf))
    return jnp.min(masked_priorities) / total_priority


def init_per_buffer(prototype: object, capacity: int) -> PERBufferState:
    storage_capacity = _next_power_of_two(capacity)
    leaves, _ = jax.tree.flatten(prototype)

    flat_data = {}
    for i, leaf in enumerate(leaves):
        flat_data[str(i)] = jnp.zeros((storage_capacity, *leaf.shape), dtype=leaf.dtype)

    return PERBufferState(
        data=flat_data,
        pointer=jnp.array(0, dtype=jnp.uint32),
        count=jnp.array(0, dtype=jnp.uint32),
        logical_capacity=jnp.array(capacity, dtype=jnp.uint32),
        storage_capacity=jnp.array(storage_capacity, dtype=jnp.uint32),
        tree=tree_init(storage_capacity),
        max_priority=jnp.float32(1.0),
    )


def per_add(state: PERBufferState, transition: object, alpha: float = 0.6) -> PERBufferState:
    leaves, _ = jax.tree.flatten(transition)
    logical_capacity = state.logical_capacity.astype(jnp.uint32)
    idx = state.pointer % logical_capacity

    new_data = {}
    for i, leaf in enumerate(leaves):
        key = str(i)
        new_data[key] = state.data[key].at[idx].set(leaf)

    priority = state.max_priority**alpha
    new_tree = tree_update(state.tree, idx, priority)

    return state._replace(
        data=new_data,
        pointer=state.pointer + jnp.uint32(1),
        count=jnp.minimum(state.count + jnp.uint32(1), logical_capacity),
        tree=new_tree,
    )


def per_sample[T](
    state: PERBufferState,
    key: jax.Array,
    batch_size: int,
    beta: float,
    prototype: T,
) -> tuple[T, jax.Array, jax.Array]:
    storage_capacity = state.tree.shape[0] // 2

    indices = tree_sample_batch(state.tree, key, storage_capacity, batch_size)

    # Compute DeepMind-style normalized IS weights using the minimum non-zero
    # sampling probability among populated entries so that weights stay in [0, 1].
    total_priority = state.tree[1]
    leaf_priorities = state.tree[jnp.uint32(storage_capacity) + indices]
    probs = leaf_priorities / total_priority
    min_prob = _minimum_sample_probability(state, total_priority)
    weights = (probs / min_prob) ** (-jnp.float32(beta))

    # Gather transitions
    _, treedef = jax.tree.flatten(prototype)
    sampled_leaves = []
    int_indices = indices.astype(jnp.int32)
    for i in range(len(treedef.flatten_up_to(prototype))):
        sampled_leaves.append(state.data[str(i)][int_indices])

    transitions = treedef.unflatten(sampled_leaves)
    return transitions, weights, indices


def per_update_priorities(
    state: PERBufferState,
    indices: jax.Array,
    td_errors: jax.Array,
    alpha: float = 0.6,
    epsilon: float = 1e-6,
) -> PERBufferState:
    priorities = (jnp.abs(td_errors) + epsilon) ** alpha

    def _update_one(tree, idx_priority):
        idx, p = idx_priority
        return tree_update(tree, idx.astype(jnp.uint32), p), None

    new_tree, _ = jax.lax.scan(_update_one, state.tree, (indices, priorities))
    new_max = jnp.maximum(state.max_priority, jnp.max(jnp.abs(td_errors) + epsilon))

    return state._replace(tree=new_tree, max_priority=new_max)
