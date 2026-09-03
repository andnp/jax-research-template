"""Network construction for the two double-Q variants, which is all they differ in.

Their update rule is shared and gated elsewhere: end to end for each agent by
``test_rl_agents_learn_path.py``, on the real loss by
``test_rl_agents_terminal_bootstrap.py::test_double_q_loss_ignores_next_obs_on_terminal_rows``,
and on the online-network selection that makes that target double-Q rather than vanilla by
``test_rl_agents_double_q_selection.py``.
What is left here is the part none of those can separate -- the network each config selects,
the shapes it produces, and the constructor line that binds it to the agent.

Three gradient cases were deleted rather than migrated. Two rebuilt Double DQN's action
selection and loss on local arrays and asserted on their own arithmetic, never calling the
agent; the third asserted that a dueling network's parameters move under an arbitrary MSE
against random targets, which any differentiable module satisfies. All three stayed green
with both agents' training loops deleted. A fourth checked that a dueling forward pass
survives ``jax.jit``, which the learn-path gate now covers by running the whole agent
under it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from rl_agents.double_dqn import DoubleDQNAgent, DoubleDQNConfig
from rl_agents.dueling_dqn import DuelingDQNAgent, DuelingDQNConfig, DuelingQNetwork, _make_dueling_q_network
from rl_agents.q_networks import NatureQNetwork, make_q_network
from rl_components.env_protocol import EnvSpec

DUELING_MODULE = "DuelingHead_0"
"""The name Flax gives the head inside ``DuelingQNetwork``, absent from the plain MLP."""

SPEC = EnvSpec(id="toy-discrete", observation_shape=(2,), action_shape=(), num_actions=2)
"""The smallest discrete spec ``init`` accepts; only its shapes reach the parameter tree."""


def _module_names(agent: DoubleDQNAgent | DuelingDQNAgent) -> list[str]:
    """Top-level parameter modules of the network ``agent`` actually builds.

    Reached through ``init`` rather than through the network factory, because the binding
    from config to network is a separate line from the factory itself.

    Args:
        agent: A double-Q agent to initialise.

    Returns:
        The module names Flax assigned, in declaration order.
    """
    params = agent.init(jax.random.key(0), SPEC).train_state.params
    return list(params["params"])


class TestDoubleDQNNetwork:
    def test_double_dqn_can_use_nature_preset(self) -> None:
        net = make_q_network(
            DoubleDQNConfig(NETWORK_PRESET="nature_cnn"),
            action_dim=3,
            observation_shape=(4, 84, 84, 1),
        )
        assert isinstance(net, NatureQNetwork)
        params = net.init(jax.random.key(0), jnp.zeros((4, 84, 84, 1), dtype=jnp.uint8))
        q = net.apply(params, jnp.zeros((2, 4, 84, 84, 1), dtype=jnp.uint8))
        assert q.shape == (2, 3)


class TestDuelingDQNNetwork:
    def test_dueling_network_output_shape(self) -> None:
        net = DuelingQNetwork(action_dim=4)
        params = net.init(jax.random.key(0), jnp.zeros((8,)))
        q = net.apply(params, jnp.ones((8,)))
        assert q.shape == (4,)

    def test_dueling_batch_shape(self) -> None:
        net = DuelingQNetwork(action_dim=3)
        params = net.init(jax.random.key(0), jnp.zeros((4,)))
        q = net.apply(params, jnp.ones((10, 4)))
        assert q.shape == (10, 3)

    def test_dueling_network_uses_mlp_by_default(self) -> None:
        net = _make_dueling_q_network(DuelingDQNConfig(), action_dim=2)
        assert isinstance(net, DuelingQNetwork)

    def test_dueling_network_rejects_nature_preset_until_specified(self) -> None:
        with pytest.raises(ValueError, match="not yet supported"):
            _make_dueling_q_network(DuelingDQNConfig(NETWORK_PRESET="nature_cnn"), action_dim=2)


class TestAgentNetworkBinding:
    """Which network each *agent* builds, as opposed to which one its factory returns.

    ``test_dueling_network_uses_mlp_by_default`` above exercises the free function
    ``_make_dueling_q_network``; nothing exercised the constructor that hands it to
    :class:`~rl_agents.double_q.DoubleQAgent`. Rebinding that one line to
    ``make_q_network`` makes ``dueling_dqn`` literally ``double_dqn`` -- the two agents
    share everything else -- and left every test that can see either of them green, so the
    module's own stated reason to exist was uncovered.

    The initialised parameter tree names its modules, which is where the two networks are
    separable: ``double_dqn`` reports ``['Dense_0', 'Dense_1', 'Dense_2']`` and
    ``dueling_dqn`` reports ``['Dense_0', 'Dense_1', 'DuelingHead_0']``. Both directions
    are asserted, so neither agent may quietly acquire the other's network.
    """

    def test_dueling_agent_builds_the_dueling_head(self) -> None:
        assert DUELING_MODULE in _module_names(DuelingDQNAgent(DuelingDQNConfig()))

    def test_double_dqn_agent_builds_no_dueling_head(self) -> None:
        assert DUELING_MODULE not in _module_names(DoubleDQNAgent(DoubleDQNConfig()))
