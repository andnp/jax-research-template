"""Network construction for the two double-Q variants, which is all they differ in.

Their update rule is shared and gated elsewhere: end to end for each agent by
``test_rl_agents_learn_path.py``, and on the real loss by
``test_rl_agents_terminal_bootstrap.py::test_double_q_loss_ignores_next_obs_on_terminal_rows``.
What is left here is the part those two cannot separate -- the network each config selects,
and the shapes it produces.

Three gradient cases were deleted rather than migrated. Two rebuilt Double DQN's action
selection and loss on local arrays and asserted on their own arithmetic, never calling the
agent; the third asserted that a dueling network's parameters move under an arbitrary MSE
against random targets, which any differentiable module satisfies. All three stayed green
with both agents' training loops deleted. A fourth checked that a dueling forward pass
survives ``jax.jit``, which the learn-path gate now covers by running the whole agent
under it.
"""

import jax
import jax.numpy as jnp
import pytest
from rl_agents.double_dqn import DoubleDQNConfig
from rl_agents.dueling_dqn import DuelingDQNConfig, DuelingQNetwork, _make_dueling_q_network
from rl_agents.q_networks import NatureQNetwork, make_q_network


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
