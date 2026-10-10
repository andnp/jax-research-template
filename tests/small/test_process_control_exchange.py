import jax
import jax.numpy as jnp
from process_control.units.exchange import OxygenExchange, OxygenExchangeParams


def _exchange() -> OxygenExchange:
    return OxygenExchange(lambda temperature: jnp.full_like(temperature, 10.0))


def _kla(value: float) -> dict[str, jax.Array]:
    return {"kla": jnp.array(value)}


def _kelvin(celsius: float) -> jax.Array:
    return jnp.array(celsius) + 273.15


class TestOxygenExchange:
    def test_source_is_kla_times_the_saturation_deficit(self) -> None:
        """At the reference temperature the source is kLa (S_O,sat − S_O), per channel of the read."""
        source = _exchange().rates(jnp.array([4.0]), _kla(3.0), _kelvin(15.0), OxygenExchangeParams())

        assert jnp.allclose(source, jnp.array([18.0]))

    def test_no_source_at_saturation_or_without_aeration(self) -> None:
        """Neither a saturated liquid nor kLa = 0 transfers oxygen."""
        exchange = _exchange()
        params = OxygenExchangeParams()

        assert jnp.allclose(exchange.rates(jnp.array([10.0]), _kla(3.0), _kelvin(15.0), params), 0.0)
        assert jnp.allclose(exchange.rates(jnp.array([2.0]), _kla(0.0), _kelvin(15.0), params), 0.0)

    def test_kla_is_corrected_with_the_parameter_pack(self) -> None:
        """The correction uses the pack's θ and its reference temperature in °C."""
        params = OxygenExchangeParams(kla_theta=1.1, kla_reference_temperature=10.0)

        source = _exchange().rates(jnp.array([0.0]), _kla(2.0), _kelvin(12.0), params)

        assert jnp.allclose(source, jnp.array([2.0 * 1.1**2 * 10.0]))

    def test_saturation_reads_celsius(self) -> None:
        """A temperature passed in K reaches the saturation policy in °C."""
        exchange = OxygenExchange(lambda temperature: temperature)

        source = exchange.rates(jnp.array([0.0]), _kla(1.0), _kelvin(15.0), OxygenExchangeParams())

        assert jnp.allclose(source, jnp.array([15.0]))

    def test_declares_oxygen_as_its_only_channel_and_kla_as_its_signal(self) -> None:
        """The source needs one oxygen channel, read and written, and one kLa signal."""
        exchange = _exchange()

        assert [r.name for r in exchange.access.reads] == [r.name for r in exchange.access.writes] == ["O2"]
        assert exchange.signals == ("kla",)
