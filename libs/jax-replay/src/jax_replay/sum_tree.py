import jax
import jax.numpy as jnp


def tree_init(capacity: int) -> jax.Array:
    assert capacity > 0 and (capacity & (capacity - 1)) == 0, f"capacity must be a power of 2, got {capacity}"
    return jnp.zeros(2 * capacity, dtype=jnp.float32)


def tree_update(tree: jax.Array, leaf_index: jax.Array, priority: jax.Array) -> jax.Array:
    capacity = tree.shape[0] // 2
    idx = jnp.uint32(leaf_index + capacity)
    tree = tree.at[idx].set(priority)

    def _propagate(carry):
        tree, idx = carry
        idx = idx >> 1  # parent
        left = tree[idx << 1]
        right = tree[(idx << 1) | 1]
        tree = tree.at[idx].set(left + right)
        return tree, idx

    def _cond(carry):
        _, idx = carry
        return idx > 1

    tree, _ = jax.lax.while_loop(_cond, _propagate, (tree, idx))
    return tree


def tree_sample(tree: jax.Array, key: jax.Array, capacity: int) -> jax.Array:
    total = tree[1]
    target = jax.random.uniform(key, (), minval=0.0, maxval=total)
    return _tree_find(tree, target, capacity)


def _tree_find(tree: jax.Array, target: jax.Array, capacity: int) -> jax.Array:
    def _descend(carry):
        idx, target = carry
        left_val = tree[idx << 1]
        go_left = target < left_val
        idx = jnp.where(go_left, idx << 1, (idx << 1) | 1)
        target = jnp.where(go_left, target, target - left_val)
        return idx, target

    def _cond(carry):
        idx, _ = carry
        return idx < jnp.uint32(capacity)

    idx, _ = jax.lax.while_loop(_cond, _descend, (jnp.uint32(1), target))
    return idx - jnp.uint32(capacity)


def tree_sample_batch(tree: jax.Array, key: jax.Array, capacity: int, batch_size: int) -> jax.Array:
    total = tree[1]
    segment_size = total / batch_size
    keys = jax.random.split(key, batch_size)

    def _sample_segment(key_i, segment_idx):
        low = segment_size * segment_idx
        target = low + jax.random.uniform(key_i, ()) * segment_size
        return _tree_find(tree, target, capacity)

    indices = jax.vmap(_sample_segment)(keys, jnp.arange(batch_size, dtype=jnp.float32))
    return indices
