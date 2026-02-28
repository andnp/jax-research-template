"""Medium tests for jax_utils — JIT compilation of pytree ops and typed wrappers."""

import jax
import jax.numpy as jnp
from jax_utils.pytree import tree_add, tree_lerp, tree_mean, tree_norm, tree_zeros_like
from jax_utils.wrappers import typed_jit, typed_vmap


class TestPytreeOpsJIT:
    def test_tree_add_jit(self):
        a = {"w": jnp.array([1.0, 2.0])}
        b = {"w": jnp.array([3.0, 4.0])}
        result = jax.jit(tree_add)(a, b)
        assert jnp.allclose(result["w"], jnp.array([4.0, 6.0]))

    def test_tree_mean_jit(self):
        tree = {"a": jnp.array([1.0, 3.0]), "b": jnp.array([2.0, 4.0])}
        result = jax.jit(tree_mean)(tree)
        assert jnp.allclose(result, 2.5)

    def test_tree_norm_jit(self):
        tree = {"w": jnp.array([3.0, 4.0])}
        result = jax.jit(tree_norm)(tree)
        assert jnp.allclose(result, 5.0)

    def test_tree_lerp_jit(self):
        a = {"w": jnp.array([0.0])}
        b = {"w": jnp.array([10.0])}
        result = jax.jit(lambda a, b: tree_lerp(a, b, 0.3))(a, b)
        assert jnp.allclose(result["w"], jnp.array([3.0]))

    def test_tree_zeros_like_in_scan(self):
        """tree_zeros_like should work inside lax.scan."""
        tree = {"w": jnp.array([1.0, 2.0])}

        def step(carry, _):
            z = tree_zeros_like(carry)
            carry = tree_add(carry, z)
            return carry, None

        final, _ = jax.jit(lambda t: jax.lax.scan(step, t, None, length=3))(tree)
        assert jnp.allclose(final["w"], tree["w"])


class TestTypedJit:
    def test_basic(self):
        def add(a: jax.Array, b: jax.Array) -> jax.Array:
            return a + b

        jitted = typed_jit(add)
        result = jitted(jnp.array(1.0), jnp.array(2.0))
        assert float(result) == 3.0

    def test_preserves_output(self):
        def make_pair(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            return x, x * 2

        jitted = typed_jit(make_pair)
        a, b = jitted(jnp.array(3.0))
        assert float(a) == 3.0
        assert float(b) == 6.0


class TestTypedVmap:
    def test_basic(self):
        def square(x: jax.Array) -> jax.Array:
            return x**2

        vmapped = typed_vmap(square)
        result = vmapped(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, jnp.array([1.0, 4.0, 9.0]))

    def test_with_axis_name(self):
        def identity(x: jax.Array) -> jax.Array:
            return x

        vmapped = typed_vmap(identity, axis_name="batch")
        result = vmapped(jnp.array([1.0, 2.0]))
        assert jnp.allclose(result, jnp.array([1.0, 2.0]))

    def test_combined_jit_vmap(self):
        def f(x: jax.Array) -> jax.Array:
            return x + 1

        composed = typed_jit(typed_vmap(f))
        result = composed(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(result, jnp.array([2.0, 3.0, 4.0]))
