import jax.numpy as jnp
from process_control.units.exchange import OxygenExchange, OxygenExchangeParams


def _exchange() -> OxygenExchange:
    return OxygenExchange(lambda temperature: jnp.full_like(temperature, 10.0))


class TestOxygenExchange:
    def test_source_is_kla_times_the_saturation_deficit(self) -> None:
        """At the reference temperature the source is kLa (S_O,sat − S_O), per channel of the read."""
        source = _exchange().source(jnp.array([4.0]), jnp.array(3.0), jnp.array(15.0), OxygenExchangeParams())

        assert jnp.allclose(source, jnp.array([18.0]))

    def test_no_source_at_saturation_or_without_aeration(self) -> None:
        """Neither a saturated liquid nor kLa = 0 transfers oxygen."""
        exchange = _exchange()
        params = OxygenExchangeParams()

        assert jnp.allclose(exchange.source(jnp.array([10.0]), jnp.array(3.0), jnp.array(15.0), params), 0.0)
        assert jnp.allclose(exchange.source(jnp.array([2.0]), jnp.array(0.0), jnp.array(15.0), params), 0.0)

    def test_kla_is_corrected_with_the_parameter_pack(self) -> None:
        """The correction uses the pack's θ and reference temperature."""
        params = OxygenExchangeParams(kla_theta=1.1, kla_reference_temperature=10.0)

        source = _exchange().source(jnp.array([0.0]), jnp.array(2.0), jnp.array(12.0), params)

        assert jnp.allclose(source, jnp.array([2.0 * 1.1**2 * 10.0]))

    def test_access_declares_oxygen_as_the_only_read_and_write(self) -> None:
        """The source needs one oxygen channel, read and written."""
        access = OxygenExchange.access

        assert [r.name for r in access.reads] == [r.name for r in access.writes] == ["O2"]
