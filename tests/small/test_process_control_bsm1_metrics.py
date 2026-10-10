import jax.numpy as jnp
from process_control.benchmarks.bsm1_metrics import (
    EQIWeights,
    aeration_energy,
    effluent_quality_index,
    mixing_energy,
    operational_cost_index,
    pumping_energy,
)
from process_control.units.asm1 import ASM1Params, ASM1State


class TestEffluentQualityIndex:
    def test_matches_hand_computed_example(self) -> None:
        """Single-timestep EQI against a hand-computed weighted pollutant load."""
        params = ASM1Params(f_p=0.08, i_xb=0.08, i_xp=0.06)
        effluent = ASM1State.create(
            s_i=30.0, s_s=1.0, x_i=5.0, x_s=2.0, x_bh=3.0, x_ba=1.0, x_p=1.0,
            s_no=10.0, s_nh=2.0, s_nd=0.5, x_nd=0.3,
        )
        q_e = jnp.array(1000.0)

        eqi = effluent_quality_index(effluent, params, q_e)

        ss = 0.75 * (5.0 + 2.0 + 3.0 + 1.0 + 1.0)
        cod = 30.0 + 1.0 + 5.0 + 2.0 + 3.0 + 1.0 + 1.0
        bod5 = 0.25 * (1.0 + 2.0 + (1.0 - 0.08) * (3.0 + 1.0))
        tkn = 2.0 + 0.5 + 0.3 + 0.08 * (3.0 + 1.0) + 0.06 * (1.0 + 5.0)
        expected = (2.0 * ss + 1.0 * cod + 30.0 * tkn + 10.0 * 10.0 + 2.0 * bod5) * 1000.0 / 1000.0
        assert jnp.allclose(eqi, jnp.array(expected), atol=1e-6)

    def test_time_averages_over_trajectory(self) -> None:
        """EQI over a two-step trajectory is the mean of the per-step loads."""
        params = ASM1Params()
        effluent = ASM1State(
            s_i=jnp.array([30.0, 30.0]),
            s_s=jnp.array([1.0, 2.0]),
            x_i=jnp.array([5.0, 5.0]),
            x_s=jnp.array([2.0, 2.0]),
            x_bh=jnp.array([3.0, 3.0]),
            x_ba=jnp.array([1.0, 1.0]),
            x_p=jnp.array([1.0, 1.0]),
            s_o=jnp.array([0.0, 0.0]),
            s_no=jnp.array([10.0, 10.0]),
            s_nh=jnp.array([2.0, 2.0]),
            s_nd=jnp.array([0.5, 0.5]),
            x_nd=jnp.array([0.3, 0.3]),
            s_alk=jnp.array([5.0, 5.0]),
        )
        q_e = jnp.array([1000.0, 1000.0])

        eqi_traj = effluent_quality_index(effluent, params, q_e)
        eqi_step0 = effluent_quality_index(
            ASM1State.create(
                s_i=30.0, s_s=1.0, x_i=5.0, x_s=2.0, x_bh=3.0, x_ba=1.0, x_p=1.0,
                s_no=10.0, s_nh=2.0, s_nd=0.5, x_nd=0.3,
            ),
            params,
            jnp.array(1000.0),
        )
        eqi_step1 = effluent_quality_index(
            ASM1State.create(
                s_i=30.0, s_s=2.0, x_i=5.0, x_s=2.0, x_bh=3.0, x_ba=1.0, x_p=1.0,
                s_no=10.0, s_nh=2.0, s_nd=0.5, x_nd=0.3,
            ),
            params,
            jnp.array(1000.0),
        )
        assert jnp.allclose(eqi_traj, (eqi_step0 + eqi_step1) / 2.0, atol=1e-6)

    def test_custom_weights_change_result(self) -> None:
        params = ASM1Params()
        effluent = ASM1State.create(s_no=10.0)
        q_e = jnp.array(1000.0)

        default_eqi = effluent_quality_index(effluent, params, q_e)
        heavier_no_eqi = effluent_quality_index(effluent, params, q_e, weights=EQIWeights(b_no=100.0))

        assert float(heavier_no_eqi) > float(default_eqi)


class TestAerationEnergy:
    def test_matches_hand_computed_example(self) -> None:
        """AE = S_O,sat/(1.8*1000) * mean(sum_i V_i*KLa_i), KLa in d^-1."""
        kla = jnp.array([[240.0, 240.0, 84.0]])
        volumes = jnp.array([1333.0, 1333.0, 1333.0])

        ae = aeration_energy(kla, volumes, s_o_sat=8.0)

        weighted = 240.0 * 1333.0 + 240.0 * 1333.0 + 84.0 * 1333.0
        expected = 8.0 / (1.8 * 1000.0) * weighted
        assert jnp.allclose(ae, jnp.array(expected), atol=1e-6)

    def test_zero_aeration_gives_zero_energy(self) -> None:
        kla = jnp.zeros((5, 3))
        volumes = jnp.array([1333.0, 1333.0, 1333.0])
        assert float(aeration_energy(kla, volumes)) == 0.0


class TestPumpingEnergy:
    def test_matches_hand_computed_example(self) -> None:
        pe = pumping_energy(jnp.array(55338.0), jnp.array(18446.0), jnp.array(385.0))
        expected = 0.004 * 55338.0 + 0.008 * 18446.0 + 0.05 * 385.0
        assert jnp.allclose(pe, jnp.array(expected), atol=1e-6)


class TestMixingEnergy:
    def test_below_threshold_reactors_contribute(self) -> None:
        volumes = jnp.array([1000.0, 1000.0, 1333.0])
        kla = jnp.array([[0.0, 0.0, 240.0]])  # first two anoxic, third aerated

        me = mixing_energy(volumes, kla, unmixed_threshold=20.0)

        expected = 24.0 * (0.005 * 1000.0 + 0.005 * 1000.0)
        assert jnp.allclose(me, jnp.array(expected), atol=1e-6)

    def test_all_aerated_gives_zero(self) -> None:
        volumes = jnp.array([1000.0, 1000.0])
        kla = jnp.array([[240.0, 240.0]])
        assert float(mixing_energy(volumes, kla)) == 0.0


class TestOperationalCostIndex:
    def test_combines_all_terms(self) -> None:
        oci = operational_cost_index(ae=jnp.array(100.0), pe=jnp.array(10.0), sp=jnp.array(2.0), ec=jnp.array(1.0), me=jnp.array(5.0))
        assert jnp.allclose(oci, jnp.array(100.0 + 10.0 + 5.0 * 2.0 + 3.0 * 1.0 + 5.0), atol=1e-6)

    def test_defaults_to_energy_terms_only(self) -> None:
        oci = operational_cost_index(ae=jnp.array(100.0), pe=jnp.array(10.0))
        assert jnp.allclose(oci, jnp.array(110.0), atol=1e-6)
