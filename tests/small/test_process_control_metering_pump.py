import jax
import jax.numpy as jnp
import pytest
from process_control.actuators.metering_pump import MeteringPumpParams, reset, step


def _run(params: MeteringPumpParams, commands: list[float]) -> tuple[list[float], list[float]]:
    state = reset(params)
    outputs, delivered = [], []
    for command in commands:
        state, output, rate = step(state, jnp.array(command), params)
        outputs.append(float(output))
        delivered.append(float(rate))
    return outputs, delivered


class TestTurndown:
    @pytest.mark.parametrize(("turndown", "floor"), [(10.0, 0.2), (1000.0, 0.002)])
    def test_floor_is_max_output_over_turndown(self, turndown: float, floor: float) -> None:
        assert MeteringPumpParams(2.0, turndown, 0.0, 1.0).min_output == pytest.approx(floor)

    def test_command_below_half_the_floor_stops_the_pump(self) -> None:
        outputs, _ = _run(MeteringPumpParams(2.0, 10.0, 0.0, 1.0), [0.05, 0.0999])

        assert outputs == [0.0, 0.0]

    def test_command_between_half_the_floor_and_the_floor_runs_at_the_floor(self) -> None:
        outputs, _ = _run(MeteringPumpParams(2.0, 10.0, 0.0, 1.0), [0.1, 0.15])

        assert outputs == pytest.approx([0.2, 0.2])

    def test_command_in_the_stable_range_is_passed_and_above_max_is_clipped(self) -> None:
        outputs, _ = _run(MeteringPumpParams(2.0, 10.0, 0.0, 1.0), [0.7, 5.0])

        assert outputs == pytest.approx([0.7, 2.0])

    def test_wide_turndown_can_span_a_dose_that_a_narrow_one_cannot(self) -> None:
        narrow, _ = _run(MeteringPumpParams(0.9, 10.0, 0.0, 1.0), [0.001])
        wide, _ = _run(MeteringPumpParams(0.9, 1000.0, 0.0, 1.0), [0.001])

        assert narrow == [0.0]
        assert wide == pytest.approx([0.001])


class TestLineDelay:
    def test_zero_delay_delivers_the_output_immediately(self) -> None:
        outputs, delivered = _run(MeteringPumpParams(2.0, 10.0, 0.0, 1.0), [0.5, 1.0, 0.0])

        assert delivered == pytest.approx(outputs)

    def test_whole_step_delay_shifts_the_output_by_that_many_steps(self) -> None:
        _, delivered = _run(MeteringPumpParams(2.0, 10.0, 2.0, 1.0), [1.0, 1.0, 1.0, 0.0, 0.0])

        assert delivered == pytest.approx([0.0, 0.0, 1.0, 1.0, 1.0])

    def test_fractional_delay_averages_adjacent_outputs_over_the_step(self) -> None:
        """A 0.75-step delay delivers a quarter of a step input in its first step, then the rest."""
        _, delivered = _run(MeteringPumpParams(2.0, 10.0, 0.75, 1.0), [1.0, 1.0, 0.0, 0.0])

        assert delivered == pytest.approx([0.25, 1.0, 0.75, 0.0])

    def test_delivered_mass_equals_pumped_mass_once_the_line_has_drained(self) -> None:
        params = MeteringPumpParams(2.0, 10.0, 0.75, 1.0)
        commands = [0.0, 1.5, 1.5, 0.0, 0.4, 0.0, 0.0, 0.0]
        outputs, delivered = _run(params, commands)

        assert sum(delivered) == pytest.approx(sum(outputs))

    def test_a_pump_reset_at_an_output_delivers_it_from_the_first_step(self) -> None:
        params = MeteringPumpParams(2.0, 10.0, 1.5, 1.0)

        _, _, rate = step(reset(params, 1.0), jnp.array(1.0), params)

        assert rate == pytest.approx(1.0)


class TestJax:
    def test_jit_and_vmap_over_commands_match_the_eager_step(self) -> None:
        params = MeteringPumpParams(2.0, 10.0, 0.75, 1.0)
        commands = jnp.array([0.0, 0.1, 0.7, 5.0])

        def delivered(command: jax.Array) -> jax.Array:
            return step(reset(params), command, params)[2]

        batched = jax.jit(jax.vmap(delivered))(commands)

        assert batched == pytest.approx([float(delivered(command)) for command in commands])
