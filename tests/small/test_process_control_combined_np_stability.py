"""Numerical stability of the ASM2d-based combined N+P benchmark across its action space.

units/asm2d.reactor_step sub-steps its RK4 integration (see the module for the
stiffness derivation): the O2-Monod terms make the reaction Jacobian's S_O
entry huge near S_O=0, which put a single dt=0.02 h RK4 step outside the
integrator's stability region and let dissolved oxygen overshoot its physical
ceiling (s_o_sat) by 10x or more. These tests sweep the corners of the
action space over a multi-day rollout and assert every reactor state stays
finite and physically bounded.
"""

import itertools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from process_control.benchmarks.combined_np import CombinedNPConfig, make_combined_np_benchmark

_DO_INDEX = 7
_TOL = 1e-3


def _action_corners(n_dims: int) -> jax.Array:
    return jnp.array(list(itertools.product([0.0, 1.0], repeat=n_dims)))


def _rollout_zone_states(reset: Callable, step: Callable, action: jax.Array, n_steps: int, seed: int):
    state, _ = reset(jax.random.PRNGKey(seed))

    def scan(state: object, key: jax.Array):
        state, _obs, _reward, _done, _info = step(state, action, key)
        zones = jnp.stack([state.anaerobic_state, state.anoxic_state, state.aerobic_state])
        return state, zones

    def run(state: object):
        return jax.lax.scan(scan, state, jax.random.split(jax.random.PRNGKey(seed + 1), n_steps))

    _, zones = jax.jit(run)(state)
    return zones  # (n_steps, 3, N_COMPONENTS)


def _assert_finite_and_bounded(zones: jax.Array, s_o_sat: float) -> None:
    assert bool(jnp.all(jnp.isfinite(zones)))
    assert bool(jnp.all(zones >= -_TOL))
    do = zones[..., _DO_INDEX]
    assert bool(jnp.all(do <= s_o_sat + _TOL))


class TestCombinedNPStabilityAcrossActionSpace:
    def test_states_stay_finite_and_bounded_at_action_corners(self) -> None:
        """Combined N/P additionally action-controls kLa itself (up to
        kla_max=10), so this sweeps all 16 corners of the 4D action space."""
        config = CombinedNPConfig()
        reset, step = make_combined_np_benchmark(config)
        n_steps = 2 * config.steps_per_day

        for action in _action_corners(4):
            zones = _rollout_zone_states(reset, step, action, n_steps, seed=0)
            _assert_finite_and_bounded(zones, config.s_o_sat)
