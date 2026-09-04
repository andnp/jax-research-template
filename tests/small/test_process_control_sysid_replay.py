"""Small contracts for trace-driven process-control replay."""

import jax
import jax.numpy as jnp
import pytest
from process_control.sysid.replay import ReplayCase
from process_control.sysid.resampling import canonicalize
from process_control.sysid.runner import run_replay
from process_control.sysid.trace import SignalRole, SignalSeries, SignalSpec


def _signal(
    name: str,
    role: SignalRole,
    values: list[float],
    valid: list[bool] | None = None,
) -> SignalSeries:
    return SignalSeries(
        time=jnp.arange(len(values), dtype=jnp.float32),
        values=jnp.asarray(values, dtype=jnp.float32),
        spec=SignalSpec(name, role, "unit"),
        valid=None if valid is None else jnp.asarray(valid),
    )


def _trace(*, invalid_control: bool = False):
    control_valid = [False, True, True] if invalid_control else None
    return canonicalize(
        (
            _signal("slow", SignalRole.CONTROL, [1.0, 2.0, 3.0], control_valid),
            _signal("fast", SignalRole.CONTROL, [10.0, 20.0, 30.0]),
            _signal("load", SignalRole.DISTURBANCE, [100.0, 200.0, 300.0]),
        ),
        jnp.arange(3.0),
        1.0,
    )


class _ReplayEnvironment:
    def reset(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.array(0.0), jnp.array([0.0])

    def step(
        self,
        state: jax.Array,
        action: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        del key
        next_state = state + action[0]
        return (
            next_state,
            jnp.array([next_state]),
            -jnp.sum(action**2),
            jnp.array(False),
            {"realized": action[0] * 2.0},
        )


def _factory(case: ReplayCase) -> _ReplayEnvironment:
    del case
    return _ReplayEnvironment()


def test_replay_uses_named_supervisory_controls() -> None:
    """Pass selected setpoints in requested order through the replay kernel.

    Realized actuator diagnostics must remain outputs rather than replacing
    the supervisory action rows.
    """
    case = ReplayCase(
        _trace(),
        start=1,
        stop=3,
        warmup_steps=1,
        required_controls=("fast", "slow"),
        required_disturbances=("load",),
    )

    result = run_replay(case, _factory)

    assert jnp.allclose(result.arrays.controls, jnp.array([[20.0, 2.0], [30.0, 3.0]]))
    assert jnp.allclose(result.arrays.disturbances[:, 0], jnp.array([200.0, 300.0]))
    assert jnp.allclose(result.arrays.observations[:, 0], jnp.array([20.0, 50.0]))
    assert jnp.allclose(result.arrays.diagnostics["realized"], jnp.array([40.0, 60.0]))
    assert result.evaluation_start == 1
    assert result.evaluation_rows == slice(1, 2)


def test_replay_is_deterministic_for_a_seed() -> None:
    """Produce identical outputs when the same seeded case is replayed twice."""
    case = ReplayCase(_trace(), seed=42)

    first = run_replay(case, _factory)
    second = run_replay(case, _factory)

    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(jnp.array_equal, first.arrays, second.arrays)
    )


def test_replay_rejects_invalid_required_inputs() -> None:
    """Fail before simulation when a required control sample is invalid."""
    case = ReplayCase(_trace(invalid_control=True))

    with pytest.raises(ValueError, match="invalid required samples"):
        run_replay(case, _factory)


def test_replay_case_rejects_ranges_crossing_gap_segments() -> None:
    """Prevent a replay from carrying simulator state across a data gap."""
    trace = canonicalize(
        (_signal("load", SignalRole.DISTURBANCE, [1.0, 2.0, 3.0], [True, False, True]),),
        jnp.arange(3.0),
        1.0,
    )

    with pytest.raises(ValueError, match="segment boundary"):
        ReplayCase(trace)
