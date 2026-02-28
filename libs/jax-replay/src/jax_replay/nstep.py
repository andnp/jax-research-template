import jax
import jax.numpy as jnp


def compute_nstep_returns(
    rewards: jax.Array,
    dones: jax.Array,
    gamma: float,
    n: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute n-step discounted returns from a contiguous reward sequence.

    For each timestep t:
      R_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k} * prod_{j=0}^{k-1}(1 - d_{t+j})

    Args:
        rewards: shape (T,) reward sequence
        dones: shape (T,) done flags (1.0 = episode ended)
        gamma: discount factor
        n: number of steps

    Returns:
        nstep_rewards: shape (T,) n-step discounted returns
        nstep_dones: shape (T,) whether the episode ended within the n-step window
        bootstrap_indices: shape (T,) index to bootstrap from (t+n or first terminal)
    """
    length = rewards.shape[0]

    discount_powers = gamma ** jnp.arange(n, dtype=jnp.float32)

    def _compute_single(t):
        # Gather the n-step window starting at t, padded with zeros beyond end
        indices = t + jnp.arange(n)
        valid = indices < length

        window_rewards = jnp.where(valid, rewards[jnp.minimum(indices, length - 1)], 0.0)
        window_dones = jnp.where(valid, dones[jnp.minimum(indices, length - 1)], 0.0)

        # not_done_mask[k] = prod_{j=0}^{k-1}(1 - d_{t+j})
        # For k=0, the product is empty → 1.0
        cumulative_not_done = jnp.cumprod(1.0 - window_dones)
        # Shift right: [1.0, cumprod[0], cumprod[1], ..., cumprod[n-2]]
        not_done_before = jnp.concatenate([jnp.ones(1), cumulative_not_done[:-1]])

        nstep_return = jnp.sum(discount_powers * window_rewards * not_done_before)

        # Episode done within the window?
        any_done = jnp.any(window_dones > 0.0)

        # Bootstrap index: first terminal within window, or t + n
        # Find first done position (or n if none)
        done_positions = jnp.where(window_dones > 0.0, jnp.arange(n), n)
        first_done = jnp.min(done_positions)
        bootstrap_idx = jnp.minimum(t + first_done, jnp.int32(length - 1))

        # If no done in window, bootstrap from t + n (clamped)
        bootstrap_idx = jnp.where(any_done, bootstrap_idx, jnp.minimum(t + n, jnp.int32(length - 1)))

        return nstep_return, any_done.astype(jnp.float32), bootstrap_idx

    nstep_rewards, nstep_dones, bootstrap_indices = jax.vmap(_compute_single)(jnp.arange(length))
    return nstep_rewards, nstep_dones, bootstrap_indices
