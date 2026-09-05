"""Small tests for rl_components.metrics — loop metric projection and episode-return curves."""

import jax
import jax.numpy as jnp
import pytest
from rl_components.metrics import episode_return_curve, project_metrics


class TestProjectMetrics:
    def test_renames_env_prefixed_keys(self) -> None:
        """`env/`-prefixed keys are stripped to their bare name."""
        metrics = {"env/setpoint": jnp.array([1.0, 2.0])}
        projected = project_metrics(metrics)
        assert "setpoint" in projected
        assert "env/setpoint" not in projected
        assert jnp.allclose(projected["setpoint"], metrics["env/setpoint"])

    def test_renames_loop_reward(self) -> None:
        """`loop/reward` becomes the unprefixed `reward`."""
        metrics = {"loop/reward": jnp.array([1.0, 2.0])}
        projected = project_metrics(metrics)
        assert "reward" in projected
        assert "loop/reward" not in projected

    def test_extra_rename_argument(self) -> None:
        """Callers can project additional keys via `rename`."""
        metrics = {"loop/episode_return": jnp.array([0.0, 5.0])}
        projected = project_metrics(metrics, rename={"loop/episode_return": "episode_return"})
        assert "episode_return" in projected
        assert "loop/episode_return" not in projected

    def test_agent_keys_pass_through_unchanged(self) -> None:
        """Agent-owned keys, having no reserved prefix, are untouched."""
        metrics = {"actor_loss": jnp.array(0.5), "critic_loss": jnp.array(0.25)}
        projected = project_metrics(metrics)
        assert projected == metrics

    def test_other_loop_keys_keep_their_prefix(self) -> None:
        """Only `loop/reward` is renamed; the rest of the `loop/` namespace stays reachable."""
        metrics = {"loop/discount": jnp.array(0.99), "loop/episode_end": jnp.array(False)}
        projected = project_metrics(metrics)
        assert projected == metrics

    def test_collision_raises_value_error(self) -> None:
        """Two namespaces claiming one flat name is an error naming that name."""
        metrics = {"loop/reward": jnp.array(1.0), "reward": jnp.array(2.0)}
        with pytest.raises(ValueError, match="reward"):
            project_metrics(metrics)


class TestEpisodeReturnCurve:
    def test_forward_fills_impulse_single_run(self) -> None:
        """A `(steps,)` impulse is held at its most recent completed-episode value."""
        metrics = {
            "loop/episode_return": jnp.array([0.0, 0.0, 3.0, 0.0, 0.0, 7.0]),
            "loop/episode_end": jnp.array([False, False, True, False, False, True]),
        }
        curve = episode_return_curve(metrics)
        assert jnp.allclose(curve, jnp.array([0.0, 0.0, 3.0, 3.0, 3.0, 7.0]))

    def test_zero_before_first_episode_end(self) -> None:
        """The value before any episode has completed is zero."""
        metrics = {
            "loop/episode_return": jnp.array([0.0, 0.0, 0.0]),
            "loop/episode_end": jnp.array([False, False, False]),
        }
        curve = episode_return_curve(metrics)
        assert jnp.allclose(curve, jnp.zeros(3))

    def test_advances_across_two_episodes(self) -> None:
        """A later, different episode return overwrites the earlier held value."""
        metrics = {
            "loop/episode_return": jnp.array([0.0, 2.0, 0.0, 9.0]),
            "loop/episode_end": jnp.array([False, True, False, True]),
        }
        curve = episode_return_curve(metrics)
        assert jnp.allclose(curve, jnp.array([0.0, 2.0, 2.0, 9.0]))

    def test_output_shape_matches_input_shape(self) -> None:
        """The curve has the same shape as the input, unlike a `flatnonzero` mask."""
        metrics = {
            "loop/episode_return": jnp.array([0.0, 1.0, 0.0, 4.0]),
            "loop/episode_end": jnp.array([False, True, False, True]),
        }
        curve = episode_return_curve(metrics)
        assert curve.shape == metrics["loop/episode_return"].shape

    def test_batched_seed_axis(self) -> None:
        """A `(seeds, steps)` input holds independently along the trailing time axis."""
        metrics = {
            "loop/episode_return": jnp.array([[0.0, 3.0, 0.0], [0.0, 0.0, 5.0]]),
            "loop/episode_end": jnp.array([[False, True, False], [False, False, True]]),
        }
        curve = episode_return_curve(metrics)
        expected = jnp.array([[0.0, 3.0, 3.0], [0.0, 0.0, 5.0]])
        assert jnp.allclose(curve, expected)

    def test_under_jit(self) -> None:
        """The function is jittable as-is."""
        metrics = {
            "loop/episode_return": jnp.array([0.0, 3.0, 0.0, 7.0]),
            "loop/episode_end": jnp.array([False, True, False, True]),
        }
        curve = jax.jit(episode_return_curve)(metrics)
        assert jnp.allclose(curve, jnp.array([0.0, 3.0, 3.0, 7.0]))

    def test_under_vmap_over_seed_axis(self) -> None:
        """`vmap` over a leading seed axis reproduces the batched result."""
        returns = jnp.array([[0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
        ends = jnp.array([[False, True, False], [False, False, True]])
        def curve_fn(r: jax.Array, e: jax.Array):
            return episode_return_curve({"loop/episode_return": r, "loop/episode_end": e})

        batched = jax.vmap(curve_fn)(returns, ends)
        expected = jnp.array([[0.0, 3.0, 3.0], [0.0, 0.0, 5.0]])
        assert jnp.allclose(batched, expected)
