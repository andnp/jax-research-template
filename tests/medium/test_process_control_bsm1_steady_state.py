import jax
import jax.numpy as jnp
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, BSM1PlantState, make_bsm1_benchmark


def test_bsm1_open_loop_reaches_published_steady_state() -> None:
    """Open-loop BSM1 reaches the published steady-state operating range."""
    reset, step = make_bsm1_benchmark(BSM1BenchmarkConfig())
    state, _ = reset(jax.random.PRNGKey(0))
    action = jnp.array([10.0, 3.5])

    def scan(state: BSM1PlantState, key: jax.Array):
        state, _, _, _, info = step(state, action, key)
        return state, info["s_nh_effluent"]

    def run_scan(state: BSM1PlantState):
        return jax.lax.scan(scan, state, jax.random.split(jax.random.PRNGKey(1), 48000))

    state, effluent_nh = jax.jit(run_scan)(state)
    reactor5 = state.reactor5

    assert float(effluent_nh[-1]) < 3.0
    assert 0.3 <= float(reactor5.s_o) <= 0.8
    assert 120.0 <= float(reactor5.x_ba) <= 200.0
    assert 8.0 <= float(reactor5.s_no) <= 14.0
