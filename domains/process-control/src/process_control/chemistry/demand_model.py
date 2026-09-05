import jax


def compute_demand(ammonia: jax.Array, organics: jax.Array, turbidity: jax.Array) -> jax.Array:
    return ammonia * 7.6 + organics * 0.5 + turbidity * 0.02
