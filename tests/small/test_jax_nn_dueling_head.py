"""Small (unit) tests for DuelingHead.

The recombination ``Q(s, a) = V(s) + A(s, a) - mean_a'(A(s, a'))`` is the whole of the
head, and the mean subtraction is the load-bearing half: without it the split into a value
and an advantage is unidentifiable, since any constant can move between the two streams.
:class:`TestDuelingRecombination` pins it by the two invariances that identify it rather
than by recomputing it, so the assertions cannot agree with a broken head by copying it.

The case they replace, ``TestDuelingHeadMeanSubtraction.test_advantage_mean_is_zero``,
asserted nothing about the head at all::

    centered = q - jnp.mean(q, axis=-1, keepdims=True)
    npt.assert_allclose(jnp.mean(centered, axis=-1), 0.0, atol=1e-6)

Subtracting a mean and then averaging yields zero for *any* array, so it passed on random
values that never touched a dueling head. Its input compounded the problem: a batch of
ones makes every row identical, which collapses the batch and action means together and
leaves the axis of the subtraction unobservable in principle. Deleting the value stream and
the mean subtraction outright -- ``return a`` -- kept the whole suite green.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.typing import VariableDict
from jax_nn.heads import DuelingHead

SEED = 42
ACTION_DIM = 4
HIDDEN_FEATURES = 32
FEATURE_DIM = 16
BATCH_SIZE = 8
SHIFT = 2.5
"""Offset applied to one stream, large enough that failing to absorb it is unmissable."""

_ADVANTAGE_KERNEL_SHAPE = (HIDDEN_FEATURES, ACTION_DIM)
"""Shape of the advantage stream's output kernel, unique in the head's parameter tree."""

_VALUE_BIAS_SHAPE = (1,)
"""Shape of the value stream's output bias, also unique: the stream is scalar."""


class TestDuelingHeadShape:
    def test_single_observation(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((16,))
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.shape == (4,)

    def test_batched_observations(self) -> None:
        model = DuelingHead(action_dim=6, hidden_features=32)
        x = jnp.ones((8, 16))
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.shape == (8, 6)


def _features() -> jax.Array:
    """A batch whose rows differ, so the action axis is distinguishable from the batch axis."""
    return jax.random.normal(jax.random.key(SEED), (BATCH_SIZE, FEATURE_DIM), dtype=jnp.float32)


def _raise_every_advantage(variables: VariableDict) -> VariableDict:
    """Add one shared scalar to all of a state's advantages, differing between states.

    Adding ``SHIFT`` to every entry of the advantage stream's output kernel adds
    ``SHIFT * sum(h)`` to each of that state's action outputs, where ``h`` is the stream's
    hidden activation. The offset is therefore identical across actions -- exactly the
    freedom the mean subtraction exists to remove -- while varying across states, which is
    what averaging over the batch instead of over the actions fails to absorb.

    Args:
        variables: The head's initialised variables.

    Returns:
        A new variable tree; ``variables`` is unchanged.
    """
    return jax.tree.map(
        lambda leaf: leaf + SHIFT if leaf.shape == _ADVANTAGE_KERNEL_SHAPE else leaf, variables
    )


def _raise_one_advantage(variables: VariableDict) -> VariableDict:
    """The same surgery on a single action's column, which the head must *not* absorb."""
    return jax.tree.map(
        lambda leaf: leaf.at[:, 0].add(SHIFT) if leaf.shape == _ADVANTAGE_KERNEL_SHAPE else leaf,
        variables,
    )


def _raise_the_value(variables: VariableDict) -> VariableDict:
    """Add ``SHIFT`` to the state value, which the head must pass through untouched."""
    return jax.tree.map(lambda leaf: leaf + SHIFT if leaf.shape == _VALUE_BIAS_SHAPE else leaf, variables)


class TestDuelingRecombination:
    """The two invariances that identify ``V + A - mean_a(A)`` without recomputing it.

    A shift shared by every advantage of a state must vanish, and a shift of the state
    value must survive on every action. Together they force the mean-subtracted form.
    Dropping the subtraction (``v + a``) lets the first shift through; dropping the value
    stream (``a - mean_a(a)``) swallows the second; averaging over the batch rather than
    over the actions leaks the per-state part of the first, since only a shift shared by
    every *state* survives that mean.
    """

    def test_a_shift_shared_by_every_advantage_leaves_q_unchanged(self) -> None:
        model = DuelingHead(action_dim=ACTION_DIM, hidden_features=HIDDEN_FEATURES)
        x = _features()
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)

        assert jnp.allclose(model.apply(_raise_every_advantage(variables), x), q, atol=1e-5)

        # Control: the surgery is potent, so the invariance above is a property of the
        # recombination rather than of an advantage stream that reaches the output weakly.
        assert not jnp.allclose(model.apply(_raise_one_advantage(variables), x), q, atol=1e-5)

    def test_a_shift_of_the_state_value_moves_every_q_by_that_amount(self) -> None:
        model = DuelingHead(action_dim=ACTION_DIM, hidden_features=HIDDEN_FEATURES)
        x = _features()
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)

        shifted = model.apply(_raise_the_value(variables), x)
        assert jnp.allclose(shifted - q, SHIFT, atol=1e-5)


class TestDuelingHeadDtype:
    def test_float32_output(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((16,), dtype=jnp.float32)
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.dtype == jnp.float32

    def test_bfloat16_output(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32, dtype=jnp.bfloat16)
        x = jnp.ones((16,), dtype=jnp.bfloat16)
        variables = model.init(jax.random.key(SEED), x)
        q = model.apply(variables, x)
        assert q.dtype == jnp.bfloat16


class TestDuelingHeadParams:
    def test_has_four_dense_layers(self) -> None:
        model = DuelingHead(action_dim=4, hidden_features=32)
        x = jnp.ones((16,))
        variables = model.init(jax.random.key(SEED), x)
        params = variables["params"]
        dense_keys = [k for k in params if k.startswith("Dense_")]
        assert len(dense_keys) == 4
