import jax
import jax.numpy as jnp
from process_control.scenarios.pulse_train import PulseTrainParams, PulseTrainState, reset, step


def _levels(params: PulseTrainParams, n_steps: int, seed: int = 0) -> jax.Array:
    def body(state: PulseTrainState, key: jax.Array) -> tuple[PulseTrainState, jax.Array]:
        return step(state, params, key)

    _, levels = jax.lax.scan(body, reset(), jax.random.split(jax.random.PRNGKey(seed), n_steps))
    return levels


class TestPulseTrain:
    def test_level_is_zero_between_pulses_and_within_amplitude_bounds_during_them(self) -> None:
        """Every non-zero level lies in [low, high); with no starts the output stays zero."""
        levels = _levels(PulseTrainParams(rate_per_step=0.05, mean_duration_steps=5.0, low=2.0, high=3.0), 5000)
        active = levels[levels != 0.0]

        assert active.size > 0
        assert jnp.all((active >= 2.0) & (active < 3.0))
        assert jnp.all(_levels(PulseTrainParams(0.0, 5.0, 2.0, 3.0), 500) == 0.0)

    def test_pulse_starts_occur_at_the_configured_rate(self) -> None:
        """With one-step pulses, the fraction of active steps equals the start probability."""
        rate = 0.02
        levels = _levels(PulseTrainParams(rate_per_step=rate, mean_duration_steps=0.0, low=1.0, high=2.0), 50_000)

        assert abs(float(jnp.mean(levels > 0.0)) - rate) < 0.004

    def test_a_started_pulse_is_visible_even_with_zero_mean_duration(self) -> None:
        """A pulse lasts at least one step, so certain starts at zero mean duration give a non-zero level every step."""
        levels = _levels(PulseTrainParams(rate_per_step=1.0, mean_duration_steps=0.0, low=1.0, high=2.0), 200)

        assert jnp.all(levels >= 1.0)

    def test_a_running_pulse_holds_its_amplitude_until_it_ends(self) -> None:
        """After a start, later steps with no new starts keep the same level for the whole duration."""
        state, first = step(reset(), PulseTrainParams(1.0, 1e6, 1.0, 5.0), jax.random.PRNGKey(0))
        no_starts = PulseTrainParams(0.0, 1e6, 1.0, 5.0)

        _, later = step(state, no_starts, jax.random.PRNGKey(1))

        assert first > 0.0
        assert later == first

    def test_mean_duration_scales_active_fraction(self) -> None:
        """Longer pulses at the same start rate keep the source active for more of the run."""
        short = _levels(PulseTrainParams(0.005, 2.0, 1.0, 2.0), 50_000)
        long = _levels(PulseTrainParams(0.005, 20.0, 1.0, 2.0), 50_000)

        assert jnp.mean(long > 0.0) > 3.0 * jnp.mean(short > 0.0)

    def test_vmap_over_keys_matches_single_runs(self) -> None:
        """The source is a pure function of its key, so batching changes nothing."""
        params = PulseTrainParams(0.1, 3.0, 1.0, 2.0)
        keys = jax.random.split(jax.random.PRNGKey(3), 4)

        def level(key: jax.Array) -> jax.Array:
            return step(reset(), params, key)[1]

        batched = jax.jit(jax.vmap(level))(keys)
        single = jnp.stack([level(key) for key in keys])

        assert jnp.array_equal(batched, single)
