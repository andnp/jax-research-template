import jax
import jax.numpy as jnp
from process_control.units.asm1 import ASM1_ACCESS, ASM1_SCHEMA, ASM1Kinetics, ASM1Params, apply_arrhenius, asm1_reaction_rates

PARAMS = ASM1Params()
REFERENCE_K = 288.15


def _reads(**values: float) -> jax.Array:
    return jnp.array([values.get(requirement.name, 0.0) for requirement in ASM1_ACCESS.reads])


def _rate(rates: jax.Array, name: str) -> jax.Array:
    return rates[[requirement.name for requirement in ASM1_ACCESS.writes].index(name)]


class TestAccess:
    def test_declares_only_the_channels_the_reactions_use(self) -> None:
        """ASM1 reads 9 channels and writes 11; the inert organics take part in no reaction."""
        reads = [requirement.name for requirement in ASM1_ACCESS.reads]
        writes = [requirement.name for requirement in ASM1_ACCESS.writes]

        assert reads == ["s_s", "x_s", "x_bh", "x_ba", "s_o", "s_no", "s_nh", "s_nd", "x_nd"]
        assert writes == ["s_s", "x_s", "x_bh", "x_ba", "x_p", "s_o", "s_no", "s_nh", "s_nd", "x_nd", "s_alk"]

    def test_requirements_carry_the_asm1_channel_bases(self) -> None:
        """Every declared channel requires the basis the ASM1 schema gives that channel."""
        for requirement in (*ASM1_ACCESS.reads, *ASM1_ACCESS.writes):
            assert requirement.basis == ASM1_SCHEMA.bases[ASM1_SCHEMA.index(requirement.name)]


class TestProcessRates:
    def test_aerobic_nitrifiers_turn_ammonia_into_nitrate(self) -> None:
        """With oxygen, ammonia and autotrophs present, ammonia is consumed and nitrate produced."""
        rates = ASM1Kinetics().rates(_reads(s_o=2.0, s_nh=5.0, x_ba=150.0, x_bh=2500.0), jnp.array(REFERENCE_K), PARAMS)

        assert _rate(rates, "s_nh") < 0.0
        assert _rate(rates, "s_no") > 0.0
        assert _rate(rates, "s_o") < 0.0

    def test_anoxic_heterotrophs_denitrify(self) -> None:
        """Without oxygen, heterotrophs grow on substrate by consuming nitrate."""
        rates = ASM1Kinetics().rates(_reads(s_s=10.0, s_no=5.0, x_bh=2500.0), jnp.array(REFERENCE_K), PARAMS)

        assert _rate(rates, "s_no") < 0.0
        assert _rate(rates, "s_s") < 0.0
        assert _rate(rates, "x_bh") > 0.0

    def test_hydrolysis_releases_substrate(self) -> None:
        """With no readily biodegradable substrate left, hydrolysis of X_S is its only source."""
        rates = ASM1Kinetics().rates(_reads(s_o=2.0, x_s=100.0, x_bh=2500.0), jnp.array(REFERENCE_K), PARAMS)

        assert _rate(rates, "s_s") > 0.0

    def test_batched_reads_along_the_last_axis(self) -> None:
        """A batch of read vectors gives one rate vector per row, each matching its unbatched evaluation."""
        batch = jnp.stack([_reads(s_o=2.0, x_bh=2500.0, s_s=5.0), _reads(s_no=4.0, x_bh=1000.0, s_s=1.0)])

        rates = asm1_reaction_rates(batch, PARAMS)

        assert rates.shape == (2, len(ASM1_ACCESS.writes))
        assert jnp.allclose(rates[1], asm1_reaction_rates(batch[1], PARAMS))


class TestTemperatureContract:
    def test_kelvin_input_reproduces_the_celsius_correction_exactly(self) -> None:
        """Rates at T + 273.15 K equal the kernel at parameters corrected to T °C, bit for bit, at 10-25 °C."""
        with jax.enable_x64(True):
            reads = jnp.array([5.0, 60.0, 2500.0, 150.0, 1.5, 6.0, 4.0, 1.0, 4.0])
            for celsius in (10.0, 15.0, 20.0, 25.0):
                kelvin = jnp.array(celsius) + 273.15

                rates = ASM1Kinetics().rates(reads, kelvin, PARAMS)

                assert jnp.array_equal(rates, asm1_reaction_rates(reads, apply_arrhenius(PARAMS, jnp.array(celsius))))

    def test_reference_temperature_applies_no_correction(self) -> None:
        """At the 15 °C reference the given parameters are used unchanged, so no pre-corrected pack is needed."""
        reads = _reads(s_o=2.0, s_nh=5.0, x_ba=150.0, x_bh=2500.0, s_s=3.0)

        assert jnp.allclose(ASM1Kinetics().rates(reads, jnp.array(REFERENCE_K), PARAMS), asm1_reaction_rates(reads, PARAMS))


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
