import dataclasses

import jax
import jax.numpy as jnp
from research_runner.config import _build_config, _merge_traced, _split_traced, _traced_paths, traced
from rl_components.structs import chex_struct


@chex_struct(frozen=True, kw_only=True)
class AgentConfig:
    lr: float = traced(1e-3)
    buffer: int = 100


@chex_struct(frozen=True)
class RunConfig:
    agent: AgentConfig
    gamma: float = traced(0.99)


class TestTracedOnChexStructs:
    def test_traced_fields_are_found_through_nested_chex_structs(self):
        """The traced marker survives chex_struct, including on a nested struct."""
        assert _traced_paths(RunConfig(agent=AgentConfig())) == ("agent.lr", "gamma")

    def test_dataclasses_replace_keeps_the_marker(self):
        """A config rebuilt with dataclasses.replace still reports its traced fields."""
        config = dataclasses.replace(RunConfig(agent=AgentConfig()), gamma=0.5)

        assert _traced_paths(config) == ("agent.lr", "gamma")
        assert _split_traced(config)[0] == RunConfig(agent=AgentConfig())

    def test_build_config_checks_types_against_chex_struct_fields(self):
        """Annotations resolve on chex_struct classes, so type errors are caught."""
        config = _build_config(RunConfig(agent=AgentConfig()), {}, {"agent.lr": 1, "agent.buffer": 5.0})

        assert config.agent == AgentConfig(lr=1.0, buffer=5)

    def test_merged_traced_values_can_be_jax_tracers_under_vmap(self):
        """The static config is shared while each vmapped run reads its own traced value."""
        static, _ = _split_traced(RunConfig(agent=AgentConfig()))
        rates = jnp.array([0.1, 0.2, 0.3])

        def read_rate(rate: jax.Array):
            return _merge_traced(static, {"agent.lr": rate}).agent.lr

        assert jnp.allclose(jax.vmap(read_rate)(rates), rates)
