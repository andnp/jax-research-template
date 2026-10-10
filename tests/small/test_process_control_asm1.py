import jax
import jax.numpy as jnp
from process_control.units.asm1 import ASM1_SCHEMA, ASM1Params, apply_arrhenius, asm1_reaction_rates

PARAMS = ASM1Params(volume=1333.0)


def _state(**values: float) -> jax.Array:
    return jnp.array([values.get(name, 0.0) for name in ASM1_SCHEMA.names])


def _rate(rates: jax.Array, name: str) -> jax.Array:
    return rates[ASM1_SCHEMA.index(name)]


class TestProcessRates:
    def test_aerobic_nitrifiers_turn_ammonia_into_nitrate(self) -> None:
        """With oxygen, ammonia and autotrophs present, ammonia is consumed and nitrate produced."""
        rates = asm1_reaction_rates(_state(s_o=2.0, s_nh=5.0, x_ba=150.0, x_bh=2500.0, s_alk=5.0), PARAMS)

        assert _rate(rates, "s_nh") < 0.0
        assert _rate(rates, "s_no") > 0.0
        assert _rate(rates, "s_o") < 0.0

    def test_anoxic_heterotrophs_denitrify(self) -> None:
        """Without oxygen, heterotrophs grow on substrate by consuming nitrate."""
        rates = asm1_reaction_rates(_state(s_s=10.0, s_no=5.0, x_bh=2500.0, s_alk=5.0), PARAMS)

        assert _rate(rates, "s_no") < 0.0
        assert _rate(rates, "s_s") < 0.0
        assert _rate(rates, "x_bh") > 0.0

    def test_hydrolysis_releases_substrate(self) -> None:
        """With no readily biodegradable substrate left, hydrolysis of X_S is its only source."""
        rates = asm1_reaction_rates(_state(s_o=2.0, x_s=100.0, x_bh=2500.0), PARAMS)

        assert _rate(rates, "s_s") > 0.0


class TestArrheniusCorrection:
    def test_reference_temperature_leaves_rates_unchanged(self) -> None:
        """At the 15 °C calibration temperature the BSM1 rates are returned as given."""
        corrected = apply_arrhenius(PARAMS, jnp.array(15.0))

        assert jnp.allclose(corrected.mu_h, PARAMS.mu_h)
        assert jnp.allclose(corrected.mu_a, PARAMS.mu_a)

    def test_rates_rise_with_temperature(self) -> None:
        """Growth runs faster in warm water and slower in cold water."""
        warm = apply_arrhenius(PARAMS, jnp.array(20.0))
        cold = apply_arrhenius(PARAMS, jnp.array(10.0))

        assert cold.mu_h < PARAMS.mu_h < warm.mu_h
        assert cold.mu_a < PARAMS.mu_a < warm.mu_a

    def test_autotrophs_are_more_temperature_sensitive(self) -> None:
        """Nitrifier growth changes more across 10-20 °C than heterotroph growth, as in BSM1_LT."""
        warm = apply_arrhenius(PARAMS, jnp.array(20.0))
        cold = apply_arrhenius(PARAMS, jnp.array(10.0))

        assert warm.mu_a / cold.mu_a > warm.mu_h / cold.mu_h

    def test_traced_temperature_under_jit(self) -> None:
        """The correction accepts a traced temperature, as a jitted plant step supplies it."""
        @jax.jit
        def corrected_mu_a(temperature: jax.Array) -> jax.Array:
            return jnp.asarray(apply_arrhenius(PARAMS, temperature).mu_a)

        assert corrected_mu_a(jnp.array(18.0)) > PARAMS.mu_a
