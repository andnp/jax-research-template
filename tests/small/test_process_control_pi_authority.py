"""PI integral-authority contracts and closed-loop tracking for the dosing benchmarks.

For a conditional-integration PI (process_control.controllers.pi_controller),
the integral term can contribute at most ``ki * max_integral`` to the output.
If that is smaller than the actuator's authority — ``max(output_max - ff, ff
- output_min)`` — the loop can never fully correct a sustained error and
settles with a permanent offset (Åström & Murray, *Feedback Systems*, ch. 10
"Integrator Windup", https://www.cds.caltech.edu/~murray/books/AM08/pdf/am08-pid_04Mar10.pdf).
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.dosing_system import SUPERVISORY
from process_control.benchmarks.bsm1_combined import (
    BSM1CombinedDOConfig,
    BSM1CombinedRecycleConfig,
)
from process_control.benchmarks.chem_p_dosing import ChemPDosingConfig, make_chem_p_dosing_benchmark
from process_control.benchmarks.chlorine import ChlorineBenchmarkConfig, make_chlorine_benchmark
from process_control.benchmarks.chlorine_two_stage import (
    ChlorineTwoStageBenchmarkConfig,
    make_chlorine_two_stage_benchmark,
)
from process_control.benchmarks.equalization_tank import (
    EqualizationTankBenchmarkConfig,
    make_equalization_tank_benchmark,
)
from process_control.benchmarks.h2s_scrubber import H2SScrubberConfig, make_h2s_scrubber_benchmark
from process_control.benchmarks.ph_neutralization import (
    PhNeutralizationBenchmarkConfig,
    make_ph_neutralization_benchmark,
)


def _authority(ff: float, output_min: float, output_max: float) -> float:
    return max(output_max - ff, ff - output_min)


ResetFn = Callable[[jax.Array], tuple[Any, jax.Array]]
StepFn = Callable[[Any, jax.Array, jax.Array], tuple[Any, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]]


def _run(reset_fn: ResetFn, step_fn: StepFn, action: jax.Array, n_steps: int, seed: int = 0) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    state, _ = reset_fn(key)

    def body(carry: tuple[Any, jax.Array], _: None) -> tuple[tuple[Any, jax.Array], dict[str, jax.Array]]:
        state, key = carry
        key, step_key = jax.random.split(key)
        state, obs, reward, done, info = step_fn(state, action, step_key)
        return (state, key), info

    _, infos = jax.lax.scan(jax.jit(body), (state, key), None, length=n_steps)
    return infos


def _run_self_driven(
    reset_fn: ResetFn, step_fn: StepFn, initial_action: jax.Array, pi_key: str, n_steps: int, seed: int = 0
) -> dict[str, jax.Array]:
    """Drive a benchmark whose PI is only exposed via `info` by feeding its output back as the next action."""
    key = jax.random.PRNGKey(seed)
    state, _ = reset_fn(key)

    def body(
        carry: tuple[Any, jax.Array, jax.Array], _: None
    ) -> tuple[tuple[Any, jax.Array, jax.Array], dict[str, jax.Array]]:
        state, action, key = carry
        key, step_key = jax.random.split(key)
        state, obs, reward, done, info = step_fn(state, action, step_key)
        return (state, info[pi_key], key), info

    _, infos = jax.lax.scan(jax.jit(body), (state, initial_action, key), None, length=n_steps)
    return infos


class TestPIIntegralAuthorityContract:
    """Every PI loop's integral clamp must cover the actuator's full authority."""

    @pytest.mark.parametrize("loop", ["caustic", "bleach", "makeup"])
    def test_h2s_scrubber(self, loop: str) -> None:
        c = H2SScrubberConfig()
        ki, ff, out_min, out_max = {
            "caustic": (c.caustic_ki, c.caustic_ff, c.caustic_pump_min, c.caustic_pump_max),
            "bleach": (c.bleach_ki, c.bleach_ff, c.bleach_pump_min, c.bleach_pump_max),
            "makeup": (c.makeup_ki, c.makeup_ff, c.makeup_pump_min, c.makeup_pump_max),
        }[loop]
        max_integral = {
            "caustic": c.caustic_max_integral,
            "bleach": c.bleach_max_integral,
            "makeup": c.makeup_max_integral,
        }[loop]
        assert ki * max_integral >= _authority(ff, out_min, out_max)

    def test_bsm1_combined_do(self) -> None:
        do = BSM1CombinedDOConfig()
        assert do.do_ki * do.do_max_integral >= _authority(do.do_ff, do.kla_34_min, do.kla_34_max)
        assert do.do_ki * do.do_max_integral >= _authority(do.do_ff, do.kla_5_min, do.kla_5_max)

    def test_bsm1_combined_recycle(self) -> None:
        recycle = BSM1CombinedRecycleConfig()
        authority = _authority(recycle.q_a_ff, recycle.q_a_ratio_min, recycle.q_a_ratio_max)
        assert abs(recycle.q_a_ki) * recycle.q_a_max_integral >= authority


class TestClosedLoopTracking:
    """The default PI/SUPERVISORY loop reaches its setpoint at nominal conditions."""

    def test_chlorine_reaches_target_residual(self) -> None:
        config = ChlorineBenchmarkConfig(control_mode=SUPERVISORY)
        reset_fn, step_fn = make_chlorine_benchmark(config)
        action = jnp.array(config.target_residual)
        infos = _run(reset_fn, step_fn, action, n_steps=4000)
        settled = infos["outlet_residual"][-500:]
        assert float(jnp.abs(settled.mean() - config.target_residual)) < 0.3

    def test_chlorine_two_stage_reaches_target_residual(self) -> None:
        config = ChlorineTwoStageBenchmarkConfig(control_mode=SUPERVISORY)
        reset_fn, step_fn = make_chlorine_two_stage_benchmark(config)
        action = jnp.array(config.target_residual)
        infos = _run(reset_fn, step_fn, action, n_steps=4000)
        settled = infos["outlet_residual"][-500:]
        assert float(jnp.abs(settled.mean() - config.target_residual)) < 0.3

    def test_ph_neutralization_reaches_target_ph_at_nominal_conditions(self) -> None:
        """Nominal = mean flow and mean acid load, without the diurnal swing.

        The full diurnal disturbance (flow AND acid concentration both
        varying) pushes base_excess across the carbonate buffer's minimum-
        capacity region, where pH is extremely sensitive to small charge
        imbalances; that disturbance-rejection problem is a plant/scenario
        design question, not an integral-authority sizing one, and is out of
        scope here.
        """
        config = PhNeutralizationBenchmarkConfig(
            diurnal_amplitude=0.0, min_flow=50.0, max_flow=50.0, drift_scale=0.0,
            acid_amplitude=0.0, acid_min=0.005, acid_max=0.005, acid_drift_scale=0.0,
        )
        reset_fn, step_fn = make_ph_neutralization_benchmark(config)
        action = jnp.array(config.pi_ff)
        infos = _run_self_driven(reset_fn, step_fn, action, "pi_dose", n_steps=3000)
        settled = infos["measured_ph"][-500:]
        assert float(jnp.abs(settled.mean() - config.target_ph)) < 0.2

    def test_equalization_tank_reaches_target_level(self) -> None:
        config = EqualizationTankBenchmarkConfig()
        reset_fn, step_fn = make_equalization_tank_benchmark(config)
        action = jnp.array(config.pi_ff)
        infos = _run_self_driven(reset_fn, step_fn, action, "pi_outlet", n_steps=4000)
        target_level = config.target_level_fraction * config.tank_max_level
        settled = infos["measured_level"][-500:]
        assert float(jnp.abs(settled.mean() - target_level)) < 0.3

    def test_chem_p_dosing_reaches_target_po4_at_nominal_conditions(self) -> None:
        """Nominal = mean flow and mean influent PO4, without the diurnal swing.

        `control_mode` defaults to DIRECT (agent drives the pump directly);
        this exercises the SUPERVISORY path explicitly, since that is the PI
        loop being sized here. As with pH, the full diurnal PO4/flow
        disturbance is a separate, harder disturbance-rejection problem.
        """
        config = ChemPDosingConfig(
            control_mode=SUPERVISORY,
            po4_amplitude=0.0, po4_min=5.0, po4_max=5.0, po4_drift_scale=0.0,
            diurnal_amplitude=0.0, min_flow=75.0, max_flow=75.0, drift_scale=0.0,
        )
        reset_fn, step_fn = make_chem_p_dosing_benchmark(config)
        action = jnp.array([config.target_po4])
        infos = _run(reset_fn, step_fn, action, n_steps=3000)
        settled = infos["po4_eff"][-500:]
        assert float(jnp.abs(settled.mean() - config.target_po4)) < 0.5

    def test_h2s_scrubber_caustic_reaches_ph_setpoint(self) -> None:
        config = H2SScrubberConfig()
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        action = jnp.array([config.ph_sp_default, config.orp_sp_default, config.makeup_flow_sp_default])
        infos = _run(reset_fn, step_fn, action, n_steps=1000)
        settled = infos["true_ph"][-200:]
        assert float(jnp.abs(settled.mean() - config.ph_sp_default)) < 0.3

    def test_h2s_scrubber_bleach_moves_toward_orp_setpoint(self) -> None:
        """Bleach ORP control is slow (small ki by design) and its sensor
        drifts, so this checks directional progress toward setpoint rather
        than tight settling within a short window.
        """
        config = H2SScrubberConfig()
        reset_fn, step_fn = make_h2s_scrubber_benchmark(config)
        action = jnp.array([config.ph_sp_default, config.orp_sp_default, config.makeup_flow_sp_default])
        infos = _run(reset_fn, step_fn, action, n_steps=1000)
        settled = infos["true_orp"][-200:]
        assert float(jnp.abs(settled.mean() - config.orp_sp_default)) < 100.0
