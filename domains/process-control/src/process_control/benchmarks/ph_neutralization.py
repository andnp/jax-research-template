"""Two-stage pH neutralisation of an acidic, bicarbonate-buffered wastewater with caustic (NaOH).

Two CSTRs in series, coarse then trim, with a metering pump dosing NaOH into each; the action is the two
pump commands (mol/min). pH comes from the carbonate speciation solver (`chemistry/speciation.py`).

Units: volume m3, time min, flow L/min (the unit `disturbance_events` rain storms act in; tank flow is
flow / 1000), S in eq/m3 (= meq/L; positive for net strong-cation excess), Ct (total carbon) in mol/m3 (= mmol/L).

Reaction invariants per tank are [S, Ct]; mixing and dosing are linear in them (Gustafsson & Waller 1983,
https://doi.org/10.1016/0009-2509(83)80157-2; Henson & Seborg 1994, https://doi.org/10.1109/87.317975), so
`units/cstr.step` is reused unchanged on the stacked vector. One mole of NaOH adds one equivalent to S.

Influent source mapping: the buffer stream is NaHCO3 at the hidden alkalinity Alk (eq/m3 = mol/m3 NaHCO3)
plus strong mineral acid, so S_in = Alk - acid and Ct_in = Alk. Alkalinity is not added a second time as a
strong-ion term. (The retired model's `buffer_capacity = 0.005 mol/L` was 5 mmol/L total carbon, not 5 meq/L;
`Ct = 1000 * buffer_capacity` was a total-carbon mapping.) 1 mg/L as CaCO3 = 0.02 eq/m3.

pH model change from the retired `chemistry/ph_model.py`, which this replaces: constants are the T-dependent
`aqueous` carbonate values (pKa1 6.352, pKa2 10.329, pKw 13.995 at 298.15 K) instead of pKa1 6.35, pKa2 10.25,
pKw 14. The two are not numerically equivalent.

Tank 2 takes tank 1's state from the start of the step (one-step explicit coupling): adequate at dt = 1 min
against a 12 min tank-1 time constant.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from process_control._jax_dataclass import jax_dataclass
from process_control.actuators import metering_pump as pump
from process_control.chemistry.aqueous import DEFAULT_REGISTRY
from process_control.chemistry.speciation import SolveControls, evaluate_constants, fractions, make_speciation_system, solve_ideal_charge_balance
from process_control.disturbances.schedule import (
    DisturbanceSchedule,
    add_event,
    apply_active,
    create_empty,
    validate_disturbance_events,
)
from process_control.scenarios import pulse_train as pulses
from process_control.scenarios.sinusoidal_channel import ChannelParams, channel_step
from process_control.sensors import flow_sensor
from process_control.sensors import response_time as ph_sensor
from process_control.transport import BulkProperties, Composition, Hydraulics, Transport
from process_control.units.cstr import CSTRParams, CSTRState
from process_control.units.cstr import step as cstr_step

NAOH_MOLAR_MASS_KG_PER_MOL = 0.040
HOURS_PER_MIN = 1.0 / 60.0


@dataclass(frozen=True)
class PhNeutralizationBenchmarkConfig:
    """Every value marked ASSUMED is synthetic and has no field calibration."""

    dt: float = 1.0  # min
    steps_per_day: int = 1440
    temperature_k: float = 298.15  # ASSUMED
    # Covered, non-aerated tanks (ASSUMED): carbon is conserved, with no CO2 stripping.

    # Two tanks in a 20 min total residence time at the mean flow (ASSUMED split; EPA design range 10-30 min).
    volume_1: float = 0.6  # m3
    volume_2: float = 0.4  # m3

    # Flow (L/min). Flow and acid drifts are independent random walks. ALL ASSUMED.
    mean_flow: float = 50.0
    flow_amplitude: float = 8.0
    min_flow: float = 25.0
    max_flow: float = 90.0
    flow_drift_scale: float = 0.2
    flow_drift_clip: float = 10.0
    flow_slug: pulses.PulseTrainParams = pulses.PulseTrainParams(1.0 / 1440, 20.0, 15.0, 30.0)  # L/min added after clipping; ~1 per day
    flow_obs_scale: float = 120.0  # max_flow + largest slug; the observation is not clipped, so events can exceed 1

    # Strong mineral acid in the influent (eq/m3), ASSUMED.
    acid_mean: float = 5.0
    acid_amplitude: float = 2.0
    acid_min: float = 1.0
    acid_max: float = 12.0
    acid_drift_scale: float = 0.01
    acid_drift_clip: float = 1.5
    acid_slug: pulses.PulseTrainParams = pulses.PulseTrainParams(1.0 / 1440, 20.0, 3.0, 8.0)

    # Hidden NaHCO3 alkalinity (eq/m3 = mol/m3), a slow random walk. ASSUMED: 3 eq/m3 is about 150 mg/L as CaCO3.
    alk_mean: float = 3.0
    alk_min: float = 0.5
    alk_max: float = 8.0
    alk_drift_scale: float = 0.05
    alk_drift_clip: float = 2.5

    # NaOH metering pumps (mol/min), ASSUMED: 1.5x the calculated peak duty (influent to pH 6 in stage 1,
    # pH 6 to 7.5 in stage 2, over alkalinity 0.5-8, acid 1-12 and 120 L/min): 1.40 and 0.60 mol/min.
    pump_1_max: float = 2.1
    pump_2_max: float = 0.9
    # Digital diaphragm dosing pumps list 1:1000 turndown (Grundfos DDA 17-7,
    # https://www.lenntech.com/grundfos/DDAFAM/97974134/DDA-17-7-WITH-ACCESS.html), a vendor nominal range,
    # not a verified accuracy limit. Peak duty is about 10x nominal duty, so 10:1 (Neptune mechanical
    # metering pumps, https://www.psgdover.com/docs/default-source/neptune/spec-sheets/chemical-feed-system-metering-pump-specifications-for-general-use.pdf)
    # cannot span it and is the labelled stress setting.
    turndown: float = 1000.0
    dose_line_delay: float = 0.75  # min, ASSUMED

    # Electrode plus sample loop (ASSUMED; a clean electrode alone responds in seconds), flow meter about 2 %.
    ph_t90_min: float = 1.0
    ph_noise_std: float = 0.05
    flow_noise_std: float = 1.0  # L/min

    # Reward, from the TRUE tank-2 pH. The band is illustrative: permits are site-specific, and the federal
    # general pretreatment prohibition is pH below 5 (40 CFR 403.5). The margin band is reported, not rewarded.
    band_low: float = 6.0
    band_high: float = 9.0
    margin_low: float = 6.5
    margin_high: float = 8.5
    band_penalty: float = 0.5  # $ per (pH unit)^2 per minute outside the band, ASSUMED
    naoh_price_per_kg: float = 0.8  # active NaOH; Concord NC bid 2678 (June 2025), $665-730 per dry ton

    solver_iterations: int = 20  # 12 suffice over the operating range
    # Float32 cannot resolve the default 1e-4 pH / relative residual at pH ~12 with S ~ 13 eq/m3 (it flags a
    # converged step as not converged intermittently); 1e-3 is still 50x below the electrode noise.
    solver_tolerance: float = 1e-3
    max_disturbance_events: int = 16
    disturbance_events: tuple[tuple[int, int, float, int], ...] = ()


@jax_dataclass
class PhPlantState:
    step_count: jax.Array
    flow_drift: jax.Array
    acid_drift: jax.Array
    alk_drift: jax.Array
    flow_slug: pulses.PulseTrainState
    acid_slug: pulses.PulseTrainState
    tanks: CSTRState  # concentration (2, 2): [tank, (S, Ct)]
    pump_1: pump.MeteringPumpState
    pump_2: pump.MeteringPumpState
    ph_sensor_1: ph_sensor.SensorState
    ph_sensor_2: ph_sensor.SensorState
    flow_sensor: flow_sensor.FlowSensorState
    last_dose: jax.Array
    disturbance_schedule: DisturbanceSchedule


def make_ph_neutralization_benchmark(
    config: PhNeutralizationBenchmarkConfig,
) -> tuple[
    Callable[[jax.Array], tuple[PhPlantState, jax.Array]],
    Callable[[PhPlantState, jax.Array, jax.Array], tuple[PhPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]],
]:
    c = config
    max_events = validate_disturbance_events(c.max_disturbance_events, c.disturbance_events)
    system = make_speciation_system(DEFAULT_REGISTRY, ("carbonate",))
    constants = evaluate_constants(system, jnp.asarray(c.temperature_k))
    controls = SolveControls(max_iterations=c.solver_iterations, rel_tol=c.solver_tolerance, ph_tol=c.solver_tolerance)

    flow_channel = ChannelParams(c.mean_flow, c.flow_amplitude, c.min_flow, c.max_flow, 0.0, c.flow_drift_scale, c.flow_drift_clip)
    acid_channel = ChannelParams(c.acid_mean, c.acid_amplitude, c.acid_min, c.acid_max, 0.0, c.acid_drift_scale, c.acid_drift_clip)
    alk_channel = ChannelParams(c.alk_mean, 0.0, c.alk_min, c.alk_max, 0.0, c.alk_drift_scale, c.alk_drift_clip)
    pump_1_params = pump.MeteringPumpParams(c.pump_1_max, c.turndown, c.dose_line_delay, c.dt)
    pump_2_params = pump.MeteringPumpParams(c.pump_2_max, c.turndown, c.dose_line_delay, c.dt)
    ph_sensor_params = ph_sensor.SensorParams(
        ph_sensor.SensorClass(t90=c.ph_t90_min * HOURS_PER_MIN, order=2), 0.0, 14.0, c.ph_noise_std, noise_period=HOURS_PER_MIN
    )
    flow_params = flow_sensor.FlowSensorParams(c.flow_noise_std, 0.0, 0.0)
    cstr_1, cstr_2 = CSTRParams(c.volume_1), CSTRParams(c.volume_2)
    dt = jnp.asarray(c.dt)
    dt_hours = jnp.asarray(c.dt * HOURS_PER_MIN)
    max_dose = jnp.array([c.pump_1_max, c.pump_2_max])
    total_volume_l = (c.volume_1 + c.volume_2) * 1000.0

    def solve(concentration: jax.Array):
        return solve_ideal_charge_balance(system, concentration[..., 1:2], concentration[..., 0], constants, controls)

    def observe(ph_1: jax.Array, ph_2: jax.Array, flow: jax.Array, dose: jax.Array) -> jax.Array:
        return jnp.concatenate([jnp.stack([ph_1 / 14.0, ph_2 / 14.0, flow / c.flow_obs_scale]), dose / max_dose])

    def reset(rng_key: jax.Array) -> tuple[PhPlantState, jax.Array]:
        k_alk, k_s1, k_s2 = jax.random.split(rng_key, 3)
        alk_drift = jax.random.uniform(k_alk, minval=-0.5, maxval=0.5) * c.alk_drift_clip
        alk = jnp.clip(c.alk_mean + alk_drift, c.alk_min, c.alk_max)
        # Both tanks start at pH 7 for the initial alkalinity: S balances the carbonate charge, OH- and H+.
        mean_charge = jnp.sum(fractions(constants.log_ka[:2], jnp.asarray(7.0)) * jnp.array([0.0, -1.0, -2.0]))
        s_0 = -alk * mean_charge + 10.0 ** (7.0 + constants.log_kw + 3.0) - 10.0 ** (-7.0 + 3.0)
        concentration = jnp.stack([jnp.stack([s_0, alk])] * 2)
        ph_0 = solve(concentration).ph

        schedule = create_empty(max_events)
        for start_step, end_step, magnitude, type_id in c.disturbance_events:
            schedule = add_event(schedule, start_step=start_step, end_step=end_step, magnitude=magnitude, type_id=type_id)

        state = PhPlantState(
            step_count=jnp.array(0, dtype=jnp.int32),
            flow_drift=jnp.array(0.0),
            acid_drift=jnp.array(0.0),
            alk_drift=alk_drift,
            flow_slug=pulses.reset(),
            acid_slug=pulses.reset(),
            tanks=CSTRState(concentration=concentration),
            pump_1=pump.reset(pump_1_params),
            pump_2=pump.reset(pump_2_params),
            ph_sensor_1=ph_sensor.reset(ph_0[0], ph_sensor_params, k_s1),
            ph_sensor_2=ph_sensor.reset(ph_0[1], ph_sensor_params, k_s2),
            flow_sensor=flow_sensor.FlowSensorState(held_value=jnp.array(c.mean_flow)),
            last_dose=jnp.zeros(2),
            disturbance_schedule=schedule,
        )
        return state, observe(ph_0[0], ph_0[1], jnp.array(c.mean_flow), jnp.zeros(2))

    def step(
        state: PhPlantState,
        action: jax.Array,
        rng_key: jax.Array,
    ) -> tuple[PhPlantState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        k_flow, k_flow_slug, k_acid, k_acid_slug, k_alk, k_flow_sensor = jax.random.split(rng_key, 6)
        t = (state.step_count % c.steps_per_day) / c.steps_per_day

        # 1. Disturbances: flow (L/min) with slugs and configured events, acid with slugs, hidden alkalinity.
        flow_drift, flow_base = channel_step(state.flow_drift, jnp.sin(2.0 * jnp.pi * t) + jnp.cos(4.0 * jnp.pi * t), flow_channel, k_flow)
        flow_slug, flow_pulse = pulses.step(state.flow_slug, c.flow_slug, k_flow_slug)
        transport = apply_active(
            state.disturbance_schedule,
            Transport(Hydraulics(flow_base + flow_pulse), Composition.create(0.0, 0.0), BulkProperties.create(c.temperature_k - 273.15)),
            state.step_count,
        )
        flow_l_per_min = transport.hydraulics.flow
        flow = flow_l_per_min / 1000.0  # m3/min
        acid_drift, acid = channel_step(state.acid_drift, jnp.sin(2.0 * jnp.pi * t), acid_channel, k_acid)
        acid_slug, acid_pulse = pulses.step(state.acid_slug, c.acid_slug, k_acid_slug)
        acid = jnp.clip(acid + acid_pulse, c.acid_min, c.acid_max)
        alk_drift, alk = channel_step(state.alk_drift, jnp.zeros(()), alk_channel, k_alk)

        # 2. Pumps: command -> turndown -> line delay.
        pump_state_1, output_1, delivered_1 = pump.step(state.pump_1, action[0], pump_1_params)
        pump_state_2, output_2, delivered_2 = pump.step(state.pump_2, action[1], pump_2_params)

        # 3. Tanks in series on the invariants [S, Ct]; NaHCO3 influent, NaOH adds only to S.
        influent = jnp.stack([alk - acid, alk])
        tank_1 = cstr_step(CSTRState(state.tanks.concentration[0]), flow, influent, jnp.stack([delivered_1, 0.0]), cstr_1, dt)
        tank_2 = cstr_step(CSTRState(state.tanks.concentration[1]), flow, state.tanks.concentration[0], jnp.stack([delivered_2, 0.0]), cstr_2, dt)
        concentration = jnp.stack([tank_1.concentration, tank_2.concentration])

        # 4. True pH of both tanks in one batched solve.
        result = solve(concentration)
        ph = result.ph

        # 5. Sensors feed the observation only.
        ph_state_1, measured_ph_1 = ph_sensor.step(state.ph_sensor_1, ph[0], ph_sensor_params, dt_hours)
        ph_state_2, measured_ph_2 = ph_sensor.step(state.ph_sensor_2, ph[1], ph_sensor_params, dt_hours)
        flow_state, measured_flow = flow_sensor.step(state.flow_sensor, flow_l_per_min, flow_params, k_flow_sensor)

        # 6. Reward from the TRUE tank-2 pH: band violation plus reagent cost.
        dose = jnp.stack([output_1, output_2])
        violation = jnp.maximum(c.band_low - ph[1], 0.0) + jnp.maximum(ph[1] - c.band_high, 0.0)
        margin_violation = jnp.maximum(c.margin_low - ph[1], 0.0) + jnp.maximum(ph[1] - c.margin_high, 0.0)
        band_cost = c.band_penalty * violation**2 * c.dt
        naoh_cost = c.naoh_price_per_kg * NAOH_MOLAR_MASS_KG_PER_MOL * jnp.sum(dose) * c.dt
        reward = -(band_cost + naoh_cost)
        solver_valid = jnp.all(
            result.converged & result.bracket_ok & result.inputs_valid & result.model_ph_in_range & result.constants_in_domain & jnp.isfinite(ph)
        )

        new_state = PhPlantState(
            step_count=state.step_count + 1,
            flow_drift=flow_drift,
            acid_drift=acid_drift,
            alk_drift=alk_drift,
            flow_slug=flow_slug,
            acid_slug=acid_slug,
            tanks=CSTRState(concentration=concentration),
            pump_1=pump_state_1,
            pump_2=pump_state_2,
            ph_sensor_1=ph_state_1,
            ph_sensor_2=ph_state_2,
            flow_sensor=flow_state,
            last_dose=dose,
            disturbance_schedule=state.disturbance_schedule,
        )
        info: dict[str, jax.Array] = {
            "true_ph_1": ph[0],
            "true_ph_2": ph[1],
            "measured_ph_1": measured_ph_1,
            "measured_ph_2": measured_ph_2,
            "flow": flow_l_per_min,
            "hrt_min": total_volume_l / flow_l_per_min,
            "acid": acid,
            "alkalinity": alk,
            "flow_pulse": flow_pulse,
            "acid_pulse": acid_pulse,
            "event_flow": flow_l_per_min - (flow_base + flow_pulse),
            "realized_dose": dose,
            "delivered_dose": jnp.stack([delivered_1, delivered_2]),
            "band_violation": violation,
            "margin_violation": margin_violation,
            "band_cost": band_cost,
            "naoh_cost": naoh_cost,
            "solver_valid": solver_valid,
            "solver_converged": jnp.all(result.converged),
            "solver_bracket_ok": jnp.all(result.bracket_ok),
            "solver_in_domain": jnp.all(result.constants_in_domain),
        }
        return new_state, observe(measured_ph_1, measured_ph_2, measured_flow, dose), reward, jnp.array(False), info

    return reset, step
