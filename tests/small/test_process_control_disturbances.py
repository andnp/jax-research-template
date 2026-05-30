import jax
import jax.numpy as jnp
from process_control.benchmarks.chlorine import ChlorineBenchmarkConfig, make_chlorine_benchmark
from process_control.disturbances.schedule import add_event, apply_active, create_empty
from process_control.disturbances.types import (
    DISTURBANCE_DEMAND_SLUG,
    DISTURBANCE_NONE,
    DISTURBANCE_RAIN_STORM,
    apply_disturbance_type,
    demand_slug,
    rain_storm,
)
from process_control.transport import Transport


class TestDisturbanceTypes:
    def test_demand_slug_adds_to_demand(self) -> None:
        transport = Transport.create(flow=50.0, chlorine_residual=1.0, demand=0.5)
        result = demand_slug(transport, jnp.array(2.0))

        assert jnp.allclose(result.composition.ammonia, jnp.array(0.2))
        assert jnp.allclose(result.composition.turbidity, jnp.array(4.0))
        assert jnp.allclose(result.composition.organics, jnp.array(1.0))
        assert float(result.composition.demand) > float(transport.composition.demand)
        assert jnp.allclose(result.hydraulics.flow, transport.hydraulics.flow)
        assert jnp.allclose(result.composition.chlorine_residual, transport.composition.chlorine_residual)

    def test_rain_storm_increases_flow_and_demand(self) -> None:
        transport = Transport.create(flow=50.0, chlorine_residual=1.0, demand=0.5)
        result = rain_storm(transport, jnp.array(3.0))

        assert jnp.allclose(result.hydraulics.flow, jnp.array(80.0))
        assert jnp.allclose(result.composition.turbidity, jnp.array(45.0))

    def test_rain_storm_dilutes_residual(self) -> None:
        transport = Transport.create(flow=50.0, chlorine_residual=2.0, demand=0.5)
        result = rain_storm(transport, jnp.array(5.0))

        expected_residual = 2.0 * (50.0 / (50.0 + 50.0))
        assert jnp.allclose(result.composition.chlorine_residual, jnp.array(expected_residual))
        assert float(result.composition.chlorine_residual) < float(transport.composition.chlorine_residual)

    def test_apply_disturbance_type_dispatches_correctly(self) -> None:
        transport = Transport.create(flow=60.0, chlorine_residual=1.5, demand=0.3)
        mag = jnp.array(1.0)

        none_result = apply_disturbance_type(transport, jnp.array(DISTURBANCE_NONE), mag)
        assert jnp.allclose(none_result.composition.demand, transport.composition.demand)
        assert jnp.allclose(none_result.hydraulics.flow, transport.hydraulics.flow)

        slug_result = apply_disturbance_type(transport, jnp.array(DISTURBANCE_DEMAND_SLUG), mag)
        expected_slug = demand_slug(transport, mag)
        assert jnp.allclose(slug_result.composition.ammonia, expected_slug.composition.ammonia)

        storm_result = apply_disturbance_type(transport, jnp.array(DISTURBANCE_RAIN_STORM), mag)
        expected_storm = rain_storm(transport, mag)
        assert jnp.allclose(storm_result.hydraulics.flow, expected_storm.hydraulics.flow)
        assert jnp.allclose(storm_result.composition.turbidity, expected_storm.composition.turbidity)


class TestDisturbanceSchedule:
    def test_create_empty_schedule(self) -> None:
        schedule = create_empty(8)

        assert schedule.start_steps.shape == (8,)
        assert schedule.end_steps.shape == (8,)
        assert schedule.magnitudes.shape == (8,)
        assert schedule.type_ids.shape == (8,)
        assert int(schedule.count) == 0

    def test_add_event(self) -> None:
        schedule = create_empty(8)
        schedule = add_event(schedule, start_step=5, end_step=10, magnitude=2.0, type_id=DISTURBANCE_DEMAND_SLUG)

        assert int(schedule.count) == 1

    def test_active_disturbance_applied(self) -> None:
        transport = Transport.create(flow=50.0, chlorine_residual=1.0, demand=0.5)
        schedule = create_empty(8)
        schedule = add_event(schedule, start_step=0, end_step=10, magnitude=2.0, type_id=DISTURBANCE_DEMAND_SLUG)

        result = apply_active(schedule, transport, jnp.array(5, dtype=jnp.int32))

        assert float(result.composition.ammonia) > float(transport.composition.ammonia)

    def test_inactive_disturbance_not_applied(self) -> None:
        transport = Transport.create(flow=50.0, chlorine_residual=1.0, demand=0.5)
        schedule = create_empty(8)
        schedule = add_event(schedule, start_step=20, end_step=30, magnitude=2.0, type_id=DISTURBANCE_DEMAND_SLUG)

        result = apply_active(schedule, transport, jnp.array(5, dtype=jnp.int32))

        assert jnp.allclose(result.composition.ammonia, transport.composition.ammonia)

    def test_disturbance_schedule_is_jittable(self) -> None:
        config = ChlorineBenchmarkConfig()
        reset_fn, step_fn = make_chlorine_benchmark(config)

        key = jax.random.PRNGKey(0)
        state, _ = reset_fn(key)

        schedule = add_event(state.disturbance_schedule, start_step=0, end_step=100, magnitude=1.0, type_id=DISTURBANCE_RAIN_STORM)
        import dataclasses  # noqa: PLC0415
        state = dataclasses.replace(state, disturbance_schedule=schedule)

        jit_step = jax.jit(step_fn)
        k1, k2 = jax.random.split(key)
        new_state, obs, _reward, _done, _info = jit_step(state, jnp.array(2.0), k2)

        assert obs.shape == (4,)
        assert new_state.step_count == 1
