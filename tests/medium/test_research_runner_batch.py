"""One batch of configs runs as a single jit(vmap(train)) and returns each declared metric per run."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiment_definition.experiment import Experiment
from experiment_definition.metric import MetricSpec
from research_runner.batch import _run_batch
from research_runner.config import traced


@dataclass(frozen=True)
class _Toy:
    scale: float = traced(1.0)
    count: int = traced(2)
    flag: bool = traced(False)
    steps: int = 3


def _train(config: _Toy, key: jax.Array) -> tuple[object, dict[str, jax.Array]]:
    metrics = {
        "curve": jnp.arange(config.steps) * config.scale,
        "count": jnp.asarray(config.count),
        "flag": jnp.asarray(config.flag),
        "noise": jax.random.uniform(key),
    }
    return None, metrics


def _metrics(*names: str) -> tuple[MetricSpec, ...]:
    experiment = Experiment("batch")
    for name in names:
        experiment.add_metric(name)
    return experiment.metrics


def test_each_run_reads_its_own_traced_values() -> None:
    outputs = _run_batch(_train, _metrics("curve", "count"), [_Toy(scale=1.0, count=2), _Toy(scale=2.0, count=5)], [0, 1])

    np.testing.assert_allclose(outputs["curve"], [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    np.testing.assert_array_equal(outputs["count"], [2, 5])


def test_traced_values_keep_their_default_type() -> None:
    outputs = _run_batch(_train, _metrics("curve", "count", "flag"), [_Toy(flag=True), _Toy()], [0, 1])

    assert (outputs["curve"].dtype, outputs["count"].dtype, outputs["flag"].dtype) == (np.float32, np.int32, np.bool_)
    np.testing.assert_array_equal(outputs["flag"], [True, False])


def test_only_declared_metrics_are_returned() -> None:
    outputs = _run_batch(_train, _metrics("count"), [_Toy()], [0])

    assert set(outputs) == {"count"}


def test_a_run_is_reproduced_by_its_seed() -> None:
    first = _run_batch(_train, _metrics("noise"), [_Toy(), _Toy(), _Toy()], [7, 8, 7])["noise"]

    assert first[0] == first[2]
    assert first[0] != first[1]


def test_a_declared_metric_that_is_not_emitted_names_the_emitted_ones() -> None:
    with pytest.raises(ValueError, match=r"'loss' is declared but not emitted; emitted metrics are \['count', 'curve', 'flag', 'noise'\]"):
        _run_batch(_train, _metrics("loss"), [_Toy()], [0])


def test_configs_that_differ_outside_traced_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="steps differs"):
        _run_batch(_train, _metrics("count"), [_Toy(steps=3), _Toy(steps=4)], [0, 1])


def test_an_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one config"):
        _run_batch(_train, _metrics("count"), [], [])


@pytest.mark.parametrize("seed", [1.0, True])
def test_a_non_integer_seed_is_rejected(seed: int) -> None:
    with pytest.raises(TypeError, match="seeds must be integers"):
        _run_batch(_train, _metrics("count"), [_Toy()], [seed])


def test_a_seed_count_that_does_not_match_the_configs_is_rejected() -> None:
    with pytest.raises(ValueError, match="one seed per config, got 2 configs and 1 seeds"):
        _run_batch(_train, _metrics("count"), [_Toy(), _Toy()], [0])
