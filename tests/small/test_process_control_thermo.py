"""Temperature-dependent log K evaluation and its domain diagnostics."""

import jax
import jax.numpy as jnp
import pytest
from process_control.chemistry.thermo import AnalyticLogK, EquilibriumConstant, van_t_hoff


def _constant(expression: AnalyticLogK, t_min_k: float = 280.0, t_max_k: float = 320.0):
    return EquilibriumConstant(expression, "test reaction", ("test source",), t_min_k, t_max_k)


class TestAnalyticLogK:
    def test_each_coefficient_multiplies_its_own_term(self):
        """Each coefficient a1..a6 scales exactly one of 1, T, 1/T, log10 T, 1/T^2, T^2."""
        t = 300.0
        terms = (1.0, t, 1.0 / t, jnp.log10(t), 1.0 / t**2, t**2)
        for i, term in enumerate(terms):
            coefficients = [0.0] * 6
            coefficients[i] = 2.0
            value = AnalyticLogK(*coefficients)(jnp.float32(t))
            assert jnp.allclose(value, 2.0 * term, rtol=1e-5), f"coefficient a{i + 1}"

    def test_non_finite_coefficients_are_rejected(self):
        """A NaN or infinite coefficient can never produce a usable constant."""
        with pytest.raises(ValueError):
            AnalyticLogK(a1=float("nan"))
        with pytest.raises(ValueError):
            AnalyticLogK(a1=0.0, a3=float("inf"))

    def test_broadcasts_over_temperature_arrays(self):
        """Evaluation keeps the temperature array shape."""
        t = jnp.linspace(280.0, 320.0, 7)
        assert AnalyticLogK(1.0, -0.01)(t).shape == (7,)


class TestVanTHoff:
    def test_returns_reference_value_at_reference_temperature(self):
        """log K at the reference temperature equals the supplied reference value."""
        expression = van_t_hoff(-6.35, 7700.0)
        assert jnp.allclose(expression(jnp.float32(298.15)), -6.35, atol=1e-4)

    def test_matches_integrated_van_t_hoff_equation(self):
        """Away from the reference, log K follows -dH/(R ln10) (1/T - 1/Tref) in J/mol."""
        dh, r = 50_000.0, 8.314462618
        expression = van_t_hoff(-4.0, dh)
        t = 320.0
        expected = -4.0 - dh / (r * jnp.log(10.0)) * (1.0 / t - 1.0 / 298.15)
        assert jnp.allclose(expression(jnp.float32(t)), expected, atol=1e-4)

    def test_positive_enthalpy_raises_k_with_temperature(self):
        """An endothermic reaction becomes more favourable when heated."""
        expression = van_t_hoff(-4.0, 30_000.0)
        assert expression(jnp.float32(310.0)) > expression(jnp.float32(290.0))

    def test_non_positive_reference_temperature_is_rejected(self):
        """A reference temperature of zero or below is meaningless."""
        with pytest.raises(ValueError):
            van_t_hoff(-4.0, 1000.0, t_ref_k=0.0)


class TestEquilibriumConstantDomain:
    def test_inside_the_range_is_valid_and_in_domain(self):
        """Within the fitted range the value is flagged valid and in-domain."""
        result = _constant(AnalyticLogK(-5.0, 0.01)).evaluate(jnp.float32(300.0))
        assert bool(result.numerically_valid) and bool(result.in_domain)

    def test_range_endpoints_are_in_domain(self):
        """The fitted range is closed at both ends."""
        constant = _constant(AnalyticLogK(-5.0))
        assert bool(constant.evaluate(jnp.float32(280.0)).in_domain)
        assert bool(constant.evaluate(jnp.float32(320.0)).in_domain)

    def test_extrapolation_returns_the_unclamped_finite_value_but_not_in_domain(self):
        """Outside the range the expression is still evaluated at the requested temperature."""
        expression = AnalyticLogK(-5.0, 0.01)
        constant = _constant(expression)
        result = constant.evaluate(jnp.float32(350.0))
        assert bool(result.numerically_valid)
        assert not bool(result.in_domain)
        assert jnp.allclose(result.value, expression(jnp.float32(350.0)))

    @pytest.mark.parametrize("bad_t", [0.0, -10.0, float("nan"), float("inf")])
    def test_invalid_temperatures_are_not_numerically_valid(self, bad_t: float):
        """Non-positive or non-finite temperatures are flagged, never reported in-domain."""
        result = _constant(AnalyticLogK(-5.0, 0.0, 1.0)).evaluate(jnp.float32(bad_t))
        assert not bool(result.numerically_valid)
        assert not bool(result.in_domain)

    def test_diagnostics_are_elementwise_and_vmap_safe(self):
        """Batched evaluation flags each temperature independently and matches vmap."""
        constant = _constant(AnalyticLogK(-5.0, 0.01, 2.0))
        t = jnp.array([270.0, 300.0, 330.0])
        batched = constant.evaluate(t)
        mapped = jax.vmap(constant.evaluate)(t)
        assert batched.in_domain.tolist() == [False, True, False]
        assert jnp.allclose(batched.value, mapped.value)
        assert batched.in_domain.tolist() == mapped.in_domain.tolist()

    def test_jit_matches_eager(self):
        """Compiled evaluation returns the same value and flags."""
        constant = _constant(AnalyticLogK(-5.0, 0.01, 2.0))
        t = jnp.float32(305.0)
        eager, compiled = constant.evaluate(t), jax.jit(constant.evaluate)(t)
        assert jnp.allclose(eager.value, compiled.value)
        assert bool(eager.in_domain) == bool(compiled.in_domain)

    def test_malformed_records_are_rejected(self):
        """Unordered ranges and missing sources fail at construction."""
        with pytest.raises(ValueError):
            EquilibriumConstant(AnalyticLogK(0.0), "q", ("s",), 320.0, 280.0)
        with pytest.raises(ValueError):
            EquilibriumConstant(AnalyticLogK(0.0), "q", (), 280.0, 320.0)
