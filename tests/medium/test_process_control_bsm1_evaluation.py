import dataclasses
from collections.abc import Iterator

import jax
import jax.numpy as jnp
import pytest
from process_control.benchmarks.bsm1 import BSM1Handles, BSM1State, Fixed, OperatingPoint, bsm1_reference_closed_loop, make_bsm1_benchmark
from process_control.benchmarks.bsm1_metrics import EffluentLimits, evaluate

STEPS_PER_DAY = 96
WINDOW = (STEPS_PER_DAY // 2, STEPS_PER_DAY)

type Info = dict[str, jax.Array]


@pytest.fixture(scope="module", autouse=True)
def float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


@pytest.fixture(scope="module")
def dry_day(float64: None) -> Info:
    """One open-loop day of dry weather, whose diurnal load swings effluent ammonia across its limit."""
    handles = BSM1Handles(**{name: Fixed(value) for name, value in OperatingPoint().values().items()})
    reset, step = make_bsm1_benchmark(dataclasses.replace(bsm1_reference_closed_loop("dry"), handles=handles))

    def body(state: BSM1State, key: jax.Array) -> tuple[BSM1State, Info]:
        state, _, _, _, info = step(state, jnp.zeros(0), key)
        return state, info

    def run(start: BSM1State) -> Info:
        return jax.lax.scan(body, start, jax.random.split(jax.random.key(1), STEPS_PER_DAY))[1]

    return jax.jit(run)(reset(jax.random.key(0))[0])


def _window(info: Info, name: str) -> jax.Array:
    return info[name][slice(*WINDOW)]


class TestAverages:
    def test_effluent_averages_are_flow_weighted(self, dry_day: Info) -> None:
        """Each average weights the per-step flow-weighted concentrations by the effluent flow of the step."""
        report = evaluate(dry_day, *WINDOW)
        flow = _window(dry_day, "effluent_flow")

        for quality in ("s_nh", "s_no", "tss", "tkn", "n_tot", "cod", "bod5"):
            expected = jnp.sum(_window(dry_day, f"effluent_{quality}") * flow) / jnp.sum(flow)
            assert jnp.allclose(report.effluent[quality], expected, rtol=1e-9), quality

    def test_indices_average_the_window(self, dry_day: Info) -> None:
        """EQ, energies, SP and OCI over the window are the means of their per-step values, OCI included."""
        report = evaluate(dry_day, *WINDOW)
        indices = report.indices
        oci = indices["aeration_energy"] + indices["pumping_energy"] + 5.0 * indices["sludge_production"] + indices["mixing_energy"]

        assert jnp.allclose(indices["effluent_quality"], jnp.mean(_window(dry_day, "effluent_quality")))
        assert jnp.allclose(indices["operational_cost"], oci)

    def test_window_selects_the_evaluated_steps(self, dry_day: Info) -> None:
        """The night and the day of the diurnal cycle report different effluent quality."""
        night = evaluate(dry_day, 0, STEPS_PER_DAY // 2)
        day = evaluate(dry_day, *WINDOW)

        assert not jnp.allclose(night.indices["effluent_quality"], day.indices["effluent_quality"], rtol=1e-3)


class TestLimitCompliance:
    def test_reports_every_table_9_limit(self, dry_day: Info) -> None:
        """Exceedance and violations cover all five limits; the percentiles cover ammonia, total N and TSS."""
        report = evaluate(dry_day, *WINDOW)

        assert set(report.percent_time_above_limit) == {"n_tot", "cod", "s_nh", "tss", "bod5"}
        assert set(report.violations) == {"n_tot", "cod", "s_nh", "tss", "bod5"}
        assert set(report.percentile_95) == {"s_nh", "n_tot", "tss"}

    def test_limit_inside_the_diurnal_swing_is_crossed(self, dry_day: Info) -> None:
        """A limit at the window's mean ammonia is exceeded part of the time and crossed at least once."""
        ammonia = _window(dry_day, "effluent_s_nh")
        report = evaluate(dry_day, *WINDOW, limits=EffluentLimits(s_nh=float(jnp.mean(ammonia))))

        assert 0.0 < report.percent_time_above_limit["s_nh"] < 100.0
        assert report.violations["s_nh"] >= 1

    def test_limits_outside_the_range_are_never_crossed(self, dry_day: Info) -> None:
        """Ammonia always above a limit is out of compliance all the time without ever crossing it."""
        below = evaluate(dry_day, *WINDOW, limits=EffluentLimits(s_nh=0.0))
        above = evaluate(dry_day, *WINDOW, limits=EffluentLimits(s_nh=1000.0))

        assert jnp.allclose(below.percent_time_above_limit["s_nh"], 100.0)
        assert jnp.allclose(above.percent_time_above_limit["s_nh"], 0.0)
        assert below.violations["s_nh"] == 0
        assert above.violations["s_nh"] == 0

    def test_percentile_lies_in_the_upper_tail(self, dry_day: Info) -> None:
        """The 95th percentile of effluent ammonia is above its median and at most its maximum."""
        ammonia = _window(dry_day, "effluent_s_nh")
        report = evaluate(dry_day, *WINDOW)

        assert jnp.median(ammonia) < report.percentile_95["s_nh"] <= jnp.max(ammonia)
