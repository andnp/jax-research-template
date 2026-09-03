from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from process_control.environment_specs import ScenarioSpec, SignalSpec, SpecIdentity
from process_control.scenarios import (
    EventKind,
    EventSeverity,
    ForecastTrajectory,
    InitialCondition,
    InitialValue,
    RandomStream,
    ScenarioPack,
    SeedBundle,
    SignalTrajectory,
    TimedEvent,
)

FLOW = SignalSpec("influent.flow", "m3/h")
QUALITY = SignalSpec("influent.quality", "mg/L")


_DEFAULT_EVENT = TimedEvent(
    name="storm",
    kind=EventKind.DISTURBANCE,
    target=FLOW.name,
    units=FLOW.units,
    start_step=2,
    end_step=4,
    magnitude=1.5,
    severity=EventSeverity.HIGH,
    tags=("wet-weather",),
)

_DEFAULT_PACK = ScenarioPack(
    spec=ScenarioSpec(SpecIdentity("wet-weather", "1.2.0"), "1.0.0"),
    horizon_steps=10,
    simulation_time_step=0.25,
    signals=(FLOW, QUALITY),
    initial_conditions=(
        InitialCondition("nominal", (InitialValue("tank.level", "m", 2.0),), weight=1.0),
        InitialCondition("high", (InitialValue("tank.level", "m", 3.0),), weight=2.0),
    ),
    exogenous_signals=(SignalTrajectory(FLOW, jnp.arange(10.0)),),
    events=(_DEFAULT_EVENT,),
    forecasts=(ForecastTrajectory(SignalTrajectory(QUALITY, jnp.linspace(1.0, 2.0, 10)), 3),),
    severity=EventSeverity.HIGH,
    tags=("validation", "wet-weather"),
)


def _event(**overrides: Any) -> TimedEvent:
    return replace(_DEFAULT_EVENT, **overrides)


def _pack(**overrides: Any) -> ScenarioPack:
    return replace(_DEFAULT_PACK, **overrides)


def test_episode_is_reproducible_and_keeps_version_identity() -> None:
    first = _pack().instantiate(42)
    second = _pack().instantiate(42)

    assert first.spec.identity == SpecIdentity("wet-weather", "1.2.0")
    assert first.spec.schema_version == "1.0.0"
    assert first.initial_condition == second.initial_condition
    assert first.seeds.root_seed == second.seeds.root_seed == 42
    assert first.simulation_time_step == 0.25
    assert first.signals == (FLOW, QUALITY)
    assert first.initial_state() == {"tank.level": 3.0}
    np.testing.assert_array_equal(
        random.key_data(first.seeds.key(RandomStream.SCENARIO)),
        random.key_data(second.seeds.key(RandomStream.SCENARIO)),
    )
    assert float(first.exogenous_at(4)[FLOW.name]) == 4.0
    assert first.available_forecasts(2) == ()
    assert len(first.available_forecasts(3)) == 1


def test_named_random_streams_are_independent_and_addressable() -> None:
    seeds = SeedBundle.from_seed(7)
    repeated = SeedBundle.from_seed(7)

    for stream in RandomStream:
        np.testing.assert_array_equal(random.key_data(seeds.key(stream, 9)), random.key_data(repeated.key(stream, 9)))

    samples = {stream: float(random.normal(seeds.key(stream, 0))) for stream in RandomStream}
    assert len(set(samples.values())) == len(RandomStream)
    np.testing.assert_array_equal(
        random.key_data(seeds.key(RandomStream.SENSOR, 3)),
        random.key_data(seeds.key(RandomStream.SENSOR, 3)),
    )
    assert not np.array_equal(
        random.key_data(seeds.key(RandomStream.SENSOR, 3)),
        random.key_data(seeds.key(RandomStream.SENSOR, 4)),
    )


def test_repeating_event_uses_half_open_activation_windows() -> None:
    event = _event(repeat_every_steps=3, repeat_count=3)

    assert [step for step in range(10) if event.is_active(step)] == [2, 3, 5, 6, 8, 9]
    assert not event.is_active(4)
    assert event.final_end_step == 10
    assert _pack(events=(event,)).instantiate(0).active_events(5) == (event,)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: _event(start_step=4, end_step=4), "end_step"),
        (lambda: _event(repeat_every_steps=1, repeat_count=2), "cannot overlap"),
        (lambda: _event(repeat_count=2), "requires repeat_every_steps"),
        (lambda: _pack(events=(_event(start_step=9, end_step=11),)), "beyond.*horizon"),
        (
            lambda: _pack(
                events=(
                    _event(name="later", start_step=5, end_step=6),
                    _event(name="earlier", start_step=1, end_step=2),
                )
            ),
            "must be ordered",
        ),
    ],
)
def test_rejects_invalid_event_schedules(factory: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"exogenous_signals": (SignalTrajectory(FLOW, jnp.arange(9.0)),)}, "length"),
        (
            {"exogenous_signals": (SignalTrajectory(SignalSpec(FLOW.name, "L/s"), jnp.arange(10.0)),)},
            "units.*do not match",
        ),
        (
            {"events": (_event(target="unknown.flow"),)},
            "unknown signal",
        ),
    ],
)
def test_validates_signal_schema(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _pack(**overrides)


def test_rejects_steps_outside_episode_horizon() -> None:
    episode = _pack().instantiate(0)

    with pytest.raises(IndexError, match="outside episode horizon"):
        episode.active_events(10)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_rejects_invalid_episode_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed must be"):
        SeedBundle.from_seed(seed)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: InitialValue("level", "m", float("nan")),
        lambda: InitialCondition("empty", ()),
        lambda: _pack(
            initial_conditions=(
                InitialCondition("nominal", (InitialValue("tank.level", "m", 2.0),)),
                InitialCondition("incompatible", (InitialValue("tank.flow", "m3/h", 1.0),)),
            )
        ),
        lambda: SignalTrajectory(FLOW, jnp.asarray([1.0, jnp.nan])),
        lambda: _event(magnitude=float("inf")),
        lambda: _event(repeat_every_steps=0, repeat_count=2),
    ],
)
def test_rejects_nonphysical_scenario_values(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()
