"""Small tests for canonical continuous action normalization."""

from dataclasses import dataclass
from typing import cast

import chex
import jax
import jax.numpy as jnp
import pytest
from rl_components.action_normalization import ActionNormalizationWrapper
from rl_components.env_protocol import EnvProtocol, EnvReset, EnvSpec, EnvStep


@dataclass
class RecordedStep:
    action: jax.Array | None = None


class DummyContinuousEnv:
    def __init__(self, spec: EnvSpec) -> None:
        self._spec = spec

    def spec(self, params: None = None) -> EnvSpec:
        del params
        return self._spec

    def reset(self, key: chex.PRNGKey, params: None = None) -> EnvReset[jax.Array, RecordedStep]:
        del key, params
        return EnvReset(
            observation=jnp.zeros((2,), dtype=jnp.float32),
            state=RecordedStep(),
        )

    def step(
        self,
        key: chex.PRNGKey,
        state: RecordedStep,
        action: jax.Array,
        params: None = None,
    ) -> EnvStep[jax.Array, RecordedStep]:
        del key, params
        next_state = RecordedStep(action=action)
        return EnvStep(
            observation=jnp.asarray(action, dtype=jnp.float32),
            state=next_state,
            reward=jnp.array(0.0, dtype=jnp.float32),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            info={},
        )


class TestActionNormalizationWrapper:
    def test_spec_exposes_normalized_bounds_and_step_maps_back_to_native_range(self):
        env = cast(
            EnvProtocol[jax.Array, RecordedStep, jax.Array, None],
            DummyContinuousEnv(
            EnvSpec(
                id="native",
                observation_shape=(2,),
                action_shape=(2,),
                action_dtype=jnp.float32,
                action_low=jnp.array([-2.0, 0.0], dtype=jnp.float32),
                action_high=jnp.array([2.0, 10.0], dtype=jnp.float32),
            )
            ),
        )
        wrapper = ActionNormalizationWrapper(env)

        spec = wrapper.spec()
        reset = wrapper.reset(jax.random.key(0))
        transition = wrapper.step(
            jax.random.key(1),
            reset.state,
            jnp.array([-1.0, 0.5], dtype=jnp.float32),
        )

        assert spec.action_low is not None
        assert spec.action_high is not None
        assert jnp.allclose(spec.action_low, jnp.array([-1.0, -1.0], dtype=jnp.float32))
        assert jnp.allclose(spec.action_high, jnp.array([1.0, 1.0], dtype=jnp.float32))
        assert transition.state.action is not None
        assert jnp.allclose(transition.state.action, jnp.array([-2.0, 7.5], dtype=jnp.float32))
        assert jnp.allclose(transition.observation, jnp.array([-2.0, 7.5], dtype=jnp.float32))

    @pytest.mark.parametrize(
        ("spec", "message"),
        [
            pytest.param(
                EnvSpec(
                    id="discrete",
                    observation_shape=(1,),
                    action_shape=(),
                    action_dtype=jnp.int32,
                    num_actions=2,
                ),
                "continuous environment spec",
                id="discrete-env",
            ),
            pytest.param(
                EnvSpec(
                    id="missing-bounds",
                    observation_shape=(1,),
                    action_shape=(2,),
                    action_dtype=jnp.float32,
                ),
                "action_low and action_high bounds",
                id="missing-bounds",
            ),
            pytest.param(
                EnvSpec(
                    id="shape-mismatch",
                    observation_shape=(1,),
                    action_shape=(2,),
                    action_dtype=jnp.float32,
                    action_low=jnp.array([-1.0], dtype=jnp.float32),
                    action_high=jnp.array([1.0, 1.0], dtype=jnp.float32),
                ),
                "shape must match action_shape",
                id="shape-mismatch",
            ),
            pytest.param(
                EnvSpec(
                    id="invalid-ordering",
                    observation_shape=(1,),
                    action_shape=(2,),
                    action_dtype=jnp.float32,
                    action_low=jnp.array([0.5, -1.0], dtype=jnp.float32),
                    action_high=jnp.array([0.25, 1.0], dtype=jnp.float32),
                ),
                "action_low <= action_high",
                id="invalid-ordering",
            ),
        ],
    )
    def test_spec_rejects_invalid_wrapped_specs(self, spec: EnvSpec, message: str):
        wrapper = ActionNormalizationWrapper(
            cast(EnvProtocol[jax.Array, RecordedStep, jax.Array, None], DummyContinuousEnv(spec))
        )

        with pytest.raises(ValueError, match=message):
            wrapper.spec()

    def test_step_rejects_normalized_action_shape_mismatch(self):
        env = cast(
            EnvProtocol[jax.Array, RecordedStep, jax.Array, None],
            DummyContinuousEnv(
            EnvSpec(
                id="shape-check",
                observation_shape=(2,),
                action_shape=(2,),
                action_dtype=jnp.float32,
                action_low=jnp.array([-1.0, -1.0], dtype=jnp.float32),
                action_high=jnp.array([1.0, 1.0], dtype=jnp.float32),
            )
            ),
        )
        wrapper = ActionNormalizationWrapper(env)
        reset = wrapper.reset(jax.random.key(0))

        with pytest.raises(ValueError, match="normalized action shape"):
            wrapper.step(
                jax.random.key(1),
                reset.state,
                jnp.array([0.25], dtype=jnp.float32),
            )