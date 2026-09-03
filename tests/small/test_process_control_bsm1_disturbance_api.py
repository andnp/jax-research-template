from collections.abc import Callable
from dataclasses import fields
from typing import Any

import jax
import pytest
from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, make_bsm1_benchmark
from process_control.benchmarks.bsm1_combined import BSM1CombinedConfig, make_bsm1_combined_benchmark
from process_control.benchmarks.bsm1_lt import BSM1LTConfig, make_bsm1_lt_benchmark
from process_control.benchmarks.bsm1_recycle import BSM1RecycleConfig, make_bsm1_recycle_benchmark
from process_control.benchmarks.bsm1_takacs import BSM1TakacsConfig, make_bsm1_takacs_benchmark

BenchmarkFactory = Callable[[Any], tuple[Callable[..., Any], Callable[..., Any]]]


@pytest.mark.parametrize(
    ("factory", "config"),
    [
        (make_bsm1_benchmark, BSM1BenchmarkConfig()),
        (make_bsm1_lt_benchmark, BSM1LTConfig()),
        (make_bsm1_recycle_benchmark, BSM1RecycleConfig()),
        (make_bsm1_takacs_benchmark, BSM1TakacsConfig()),
        (make_bsm1_combined_benchmark, BSM1CombinedConfig()),
    ],
)
def test_bsm1_family_does_not_expose_unsupported_disturbance_schedule(
    factory: BenchmarkFactory,
    config: Any,
) -> None:
    reset, _ = factory(config)
    state, _ = reset(jax.random.PRNGKey(0))

    assert "disturbance_schedule" not in {field.name for field in fields(state)}


def test_bsm1_config_does_not_expose_unused_disturbance_capacity() -> None:
    assert "max_disturbance_events" not in {field.name for field in fields(BSM1BenchmarkConfig)}
