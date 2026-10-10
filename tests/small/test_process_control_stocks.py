"""Stock recipes and calibrated pump conversion; EPCOR figures appear only as example calibrations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike
from process_control.actuators.calibrated_pump import CalibratedPump
from process_control.chemistry.stocks import (
    StockChannels,
    StockComposition,
    caustic_from_mass_fraction,
    hypochlorite_from_assay,
)
from process_control.components import ComponentBasis, ComponentSchema, Quantity
from process_control.fields import Phase

T = 293.15
P = 101325.0
CHANNELS = StockChannels("sodium", "chloride", "chlorate", "free_chlorine", "total_carbonate")
NAMES = ("free_chlorine", "total_carbonate", "sodium", "chloride", "chlorate", "tracer")


def _schema(names: tuple[str, ...] = NAMES, tracer_unit: str = "g/m3"):
    return ComponentSchema(
        names,
        tuple(False for _ in names),
        tuple(0.0 for _ in names),
        tuple(
            ComponentBasis(tracer_unit if name == "tracer" else "mol/m3", Quantity.PROXY, name) for name in names
        ),
    )


def _bleach(**overrides: ArrayLike):
    arguments = {
        "available_cl2_g_per_l": 150.0,
        "density_kg_per_l": 1.20,
        "excess_naoh_g_per_l": 0.0,
        **overrides,
    }
    return hypochlorite_from_assay(
        channels=CHANNELS, temperature_k=T, pressure_pa=P, source="test assay", date="2026-10-10", **arguments
    )


def _concentration(stock: StockComposition, name: str):
    return float(stock.concentrations_mol_per_m3[stock.names.index(name)])


class TestHypochloriteAssay:
    def test_free_chlorine_divides_by_cl2_not_chlorine_atom_mass(self):
        """150 g/L available Cl2 is 2.116 mol/L of free chlorine; dividing by 35.45 would double it."""
        stock = _bleach()
        assert _concentration(stock, "free_chlorine") == pytest.approx(1000.0 * 150.0 / 70.90, rel=1e-5)
        assert _concentration(stock, "free_chlorine") != pytest.approx(1000.0 * 150.0 / 35.45, rel=0.1)

    def test_fresh_manufacture_prior_gives_equimolar_chloride(self):
        """Without a measured chloride, Cl2 + 2 NaOH -> NaOCl + NaCl + H2O sets chloride equal to free chlorine."""
        stock = _bleach()
        assert _concentration(stock, "chloride") == pytest.approx(_concentration(stock, "free_chlorine"), rel=1e-6)

    def test_sodium_balance_includes_every_counterion(self):
        """Na = free chlorine + chloride + chlorate + excess NaOH + 2 carbonate, in mol/m3."""
        stock = _bleach(
            excess_naoh_g_per_l=8.0, chloride_mol_per_l=2.5, chlorate_mol_per_l=0.05, carbonate_mol_per_l=0.1
        )
        expected = 150.0 / 70.90 + 2.5 + 0.05 + 8.0 / 40.0 + 2 * 0.1
        assert _concentration(stock, "sodium") == pytest.approx(1000.0 * expected, rel=1e-5)
        assert _concentration(stock, "total_carbonate") == pytest.approx(100.0, rel=1e-5)

    def test_stock_is_electroneutral(self):
        """Sodium equals the anion charge: OCl-, Cl-, ClO3-, excess OH- and CO3-- (charge two)."""
        stock = _bleach(excess_naoh_g_per_l=12.0, chlorate_mol_per_l=0.2, carbonate_mol_per_l=0.3)
        c = {name: _concentration(stock, name) for name in stock.names}
        hydroxide = 1000.0 * 12.0 / 40.0
        anions = c["free_chlorine"] + c["chloride"] + c["chlorate"] + hydroxide + 2 * c["total_carbonate"]
        assert c["sodium"] == pytest.approx(anions, rel=1e-5)

    def test_measured_sodium_overrides_the_recipe(self):
        """A measured sodium assay replaces the salt-recipe balance."""
        stock = _bleach(sodium_mol_per_l=4.0)
        assert _concentration(stock, "sodium") == pytest.approx(4000.0)

    def test_aged_stock_keeps_its_measured_chloride_and_chlorate(self):
        """Aged bleach has more chloride and chlorate than the fresh prior, which is passed through."""
        stock = _bleach(chloride_mol_per_l=3.0, chlorate_mol_per_l=0.4)
        assert _concentration(stock, "chloride") == pytest.approx(3000.0)
        assert _concentration(stock, "chlorate") == pytest.approx(400.0)

    def test_density_is_stored_in_kg_per_m3(self):
        """Density in kg/L converts to kg/m3 for mass-fraction conversions downstream."""
        assert float(_bleach().density_kg_per_m3) == pytest.approx(1200.0)

    def test_missing_carrier_role_is_rejected(self):
        """A stock needing a chlorate channel cannot be built for a carrier without one."""
        with pytest.raises(ValueError, match="chlorate"):
            hypochlorite_from_assay(
                channels=StockChannels("sodium", "chloride", None, "free_chlorine"),
                available_cl2_g_per_l=100.0,
                density_kg_per_l=1.1,
                temperature_k=T,
                pressure_pa=P,
                source="s",
                date="d",
            )


class TestCaustic:
    def _caustic(self, w: float, rho: float, **kwargs: float):
        return caustic_from_mass_fraction(
            channels=CHANNELS,
            mass_fraction=w,
            density_kg_per_l=rho,
            temperature_k=T,
            pressure_pa=P,
            source="test",
            date="2026-10-10",
            **kwargs,
        )

    @pytest.mark.parametrize(("w", "rho", "molar"), [(0.50, 1.52, 19.0), (0.25, 1.28, 8.0), (0.0, 1.0, 0.0)])
    def test_molarity_from_mass_fraction_and_density(self, w: float, rho: float, molar: float):
        """c = 1000 rho w / 40.00 mol/L: 50 wt% at 1.52 kg/L is 19 mol/L, about 25 rho w."""
        stock = self._caustic(w, rho)
        assert _concentration(stock, "sodium") == pytest.approx(1000.0 * molar, rel=1e-5, abs=1e-6)
        assert stock.names == ("sodium",)

    def test_carbonate_and_chloride_impurities_add_sodium(self):
        """Declared Na2CO3 and NaCl impurities add 2C + L sodium and appear on their channels."""
        stock = self._caustic(0.5, 1.52, carbonate_mol_per_l=0.2, chloride_mol_per_l=0.1)
        assert _concentration(stock, "sodium") == pytest.approx(1000.0 * (19.0 + 0.4 + 0.1), rel=1e-5)
        assert _concentration(stock, "total_carbonate") == pytest.approx(200.0)
        assert _concentration(stock, "chloride") == pytest.approx(100.0)

    def test_caustic_carries_no_free_chlorine(self):
        """Pure caustic leaves the free-chlorine channel zero in any carrier."""
        components = self._caustic(0.5, 1.52).components(_schema())
        assert float(components[NAMES.index("free_chlorine")]) == 0.0


class TestCarrierConversion:
    def test_components_land_on_named_channels_and_zero_elsewhere(self):
        """Channels are placed by name, whatever the carrier order, and unrelated channels stay zero."""
        stock = _bleach()
        forward = stock.components(_schema())
        reversed_names = tuple(reversed(NAMES))
        backward = stock.components(_schema(reversed_names))

        for name in stock.names:
            assert float(forward[NAMES.index(name)]) == pytest.approx(_concentration(stock, name))
            assert float(backward[reversed_names.index(name)]) == pytest.approx(_concentration(stock, name))
        assert float(forward[NAMES.index("tracer")]) == 0.0

    def test_carrier_without_a_stock_channel_is_rejected(self):
        """A stock channel missing from the carrier would lose material, so it fails."""
        with pytest.raises(ValueError, match="chlorate"):
            _bleach().components(_schema(("free_chlorine", "total_carbonate", "sodium", "chloride")))

    def test_carrier_channel_in_other_units_is_rejected(self):
        """Concentrations are mol/m3; a g/m3 channel of the same name must not receive them."""
        names = ("free_chlorine", "sodium", "chloride", "chlorate")
        schema = ComponentSchema(
            names,
            (False,) * 4,
            (0.0,) * 4,
            tuple(ComponentBasis("g/m3" if n == "sodium" else "mol/m3", Quantity.PROXY, n) for n in names),
        )
        with pytest.raises(ValueError, match="mol/m3"):
            _bleach().components(schema)

    def test_volumetric_feed_becomes_a_liquid_stream_in_cubic_metres(self):
        """65.28 L/h is 0.06528 m3/h, with unchanged mol/m3 concentrations and the stock's context."""
        stock = _bleach()
        stream = stock.feed_stream(_schema(), 65.28)

        assert stream.phase is Phase.LIQUID
        assert float(stream.flow) == pytest.approx(0.06528)
        assert np.allclose(stream.components, stock.components(_schema()))
        assert float(stream.temperature_k) == pytest.approx(T)
        assert float(stream.pressure_pa) == pytest.approx(P)

    def test_amount_rate_equals_stream_flow_times_concentration(self):
        """mol/h from L/h x mol/L matches m3/h x mol/m3, so no feed is counted at a wrong factor of 1000."""
        stock = _bleach()
        schema = _schema()
        stream = stock.feed_stream(schema, 40.0)
        rates = stock.amount_rates_mol_per_h(schema, 40.0)

        assert np.allclose(rates, stream.flow * stream.components, rtol=1e-6)
        assert float(rates[NAMES.index("free_chlorine")]) == pytest.approx(40.0 * 150.0 / 70.90, rel=1e-5)

    def test_zero_flow_delivers_no_material(self):
        """Zero volumetric flow gives zero amount rates whatever the stock strength."""
        assert not np.any(_bleach().amount_rates_mol_per_h(_schema(), 0.0))

    def test_stock_is_a_differentiable_jit_compatible_pytree(self):
        """Assay values are leaves: the free-chlorine feed rate is jittable and linear in the assay."""

        def chlorine_rate(assay: jax.Array):
            return _bleach(available_cl2_g_per_l=assay).amount_rates_mol_per_h(_schema(), 40.0)[0]

        assert float(jax.jit(chlorine_rate)(jnp.float32(150.0))) == pytest.approx(40.0 * 150.0 / 70.90, rel=1e-5)
        assert float(jax.grad(chlorine_rate)(jnp.float32(150.0))) == pytest.approx(40.0 / 70.90, rel=1e-5)


# EPCOR example calibrations in L/h per RPM.
PUMP = CalibratedPump(
    ("bleach", "caustic_duty", "caustic_standby"),
    jnp.array([1.632, 1.008, 0.576]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([60.0, 60.0, 60.0]),
)


class TestCalibratedPump:
    @pytest.mark.parametrize(
        ("head", "expected"), [("bleach", 65.28), ("caustic_duty", 40.32), ("caustic_standby", 23.04)]
    )
    def test_forty_rpm_delivers_each_heads_calibrated_volume(self, head: str, expected: float):
        """Duty and standby heads differ for the same speed, and the selection is by head name."""
        delivery = PUMP.deliver(40.0, PUMP.head_index(head))
        assert float(delivery.flow_l_per_h) == pytest.approx(expected, rel=1e-5)
        assert float(delivery.realized_rpm) == 40.0

    @pytest.mark.parametrize("command", [0.0, -5.0])
    def test_zero_or_negative_speed_delivers_nothing(self, command: float):
        """A pump commanded not to run delivers zero flow."""
        assert float(PUMP.deliver(command, 0).flow_l_per_h) == 0.0

    def test_speed_limit_acts_before_conversion(self):
        """An 80 RPM command on a 60 RPM pump delivers 60 RPM of flow while the command stays recorded."""
        delivery = PUMP.deliver(80.0, PUMP.head_index("caustic_duty"))
        assert float(delivery.command_rpm) == 80.0
        assert float(delivery.realized_rpm) == 60.0
        assert float(delivery.flow_l_per_h) == pytest.approx(60.0 * 1.008)

    def test_minimum_speed_raises_a_positive_command(self):
        """A positive command below the stable minimum runs at the minimum, not at the command."""
        pump = CalibratedPump(("a",), jnp.array([1.0]), jnp.array([10.0]), jnp.array([50.0]))
        assert float(pump.deliver(4.0, 0).realized_rpm) == 10.0
        assert float(pump.deliver(0.0, 0).realized_rpm) == 0.0

    def test_closed_interlock_stops_the_pump_regardless_of_command(self):
        """The interlock gates speed before conversion; realized speed and flow are zero but the command is kept."""
        delivery = PUMP.deliver(40.0, 0, permitted=False)
        assert float(delivery.command_rpm) == 40.0
        assert float(delivery.realized_rpm) == 0.0
        assert float(delivery.flow_l_per_h) == 0.0

    def test_recalibration_rescales_flow_without_changing_the_stock(self):
        """Calibration is a leaf: a 5 percent lower head calibration lowers flow by 5 percent for the same speed."""
        worn = CalibratedPump(PUMP.names, PUMP.liters_per_hour_per_rpm * 0.95, PUMP.min_rpm, PUMP.max_rpm)
        assert float(worn.deliver(40.0, 0).flow_l_per_h) == pytest.approx(0.95 * 65.28, rel=1e-5)

    def test_batched_commands_and_heads_under_jit(self):
        """Commands and head choices vectorize and compile, matching scalar calls."""
        commands = jnp.array([10.0, 40.0, 80.0])
        heads = jnp.array([0, 1, 2])

        def flows(c: jax.Array, h: jax.Array):
            return PUMP.deliver(c, h).flow_l_per_h

        batched = jax.jit(jax.vmap(flows))(commands, heads)
        expected = [10.0 * 1.632, 40.0 * 1.008, 60.0 * 0.576]
        assert np.allclose(batched, expected, rtol=1e-5)

    def test_pump_output_feeds_the_stock_stream(self):
        """Speed to litres per hour to cubic metres per hour: 40 RPM of bleach is 0.06528 m3/h."""
        flow_l_per_h = PUMP.deliver(40.0, PUMP.head_index("bleach")).flow_l_per_h
        stream = _bleach().feed_stream(_schema(), flow_l_per_h)
        assert float(stream.flow) == pytest.approx(0.06528, rel=1e-5)

    def test_duplicate_head_names_are_rejected(self):
        """Head names select pumps, so they must be unique."""
        with pytest.raises(ValueError, match="unique"):
            CalibratedPump(("a", "a"), jnp.ones(2), jnp.zeros(2), jnp.ones(2))
