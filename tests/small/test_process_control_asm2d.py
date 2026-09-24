from dataclasses import replace

import jax.numpy as jnp
from process_control.units.asm2d import (
    N_COMPONENTS_ASM2D,
    ASM2dParams,
    make_default_influent_asm2d,
    perfect_clarifier,
    reactions_asm2d,
)


def _typical_state() -> jnp.ndarray:
    state = make_default_influent_asm2d()
    state = state.at[4].set(2500.0)  # X_BH
    state = state.at[5].set(150.0)  # X_BA
    state = state.at[7].set(2.0)  # S_O
    state = state.at[14].set(500.0)  # X_PAO
    state = state.at[15].set(100.0)  # X_PHA
    state = state.at[16].set(100.0)  # X_PP
    return state


class TestPaoKineticTimescale:
    def test_rates_are_hourly_not_daily(self) -> None:
        """PAO rate constants must be on the same h⁻¹ clock as ASM1, not 24x faster."""
        params = ASM2dParams()
        assert params.mu_pao == 1.0 / 24.0
        assert params.q_pha == 3.0 / 24.0
        assert params.b_pao == 0.2 / 24.0

    def test_pao_growth_rate_is_bounded_per_hour(self) -> None:
        """A single hourly step should not multiply X_PAO by anything close to
        the daily rate (which the pre-fix 24x-too-fast kinetics would do)."""
        state = _typical_state()
        params = ASM2dParams()
        dc = reactions_asm2d(state, params)
        x_pao = state[14]
        assert abs(float(dc[14])) < 0.5 * float(x_pao)


class TestStorageSaturation:
    def test_storage_term_never_negative_above_kmax(self) -> None:
        """(K_MAX - f)/(K_IPP + K_MAX - f) must clip at 0, not go negative, once
        X_PP/X_PAO exceeds K_MAX. Isolate process 10 by zeroing lysis (b_pp=0)
        and anaerobic release (no S_S, so rho_9=0)."""
        params = replace(ASM2dParams(), b_pp=0.0)
        state = _typical_state()
        state = state.at[1].set(0.0)  # S_S = 0 -> rho_9 = 0
        state = state.at[16].set(0.6 * state[14])  # X_PP/X_PAO = 0.6 >> K_MAX (0.34)
        dc = reactions_asm2d(state, params)
        assert float(dc[16]) >= -1e-6


class TestConservation:
    def test_total_p_and_n_conserved_with_no_flow(self) -> None:
        """Across one reaction step (no inflow/outflow), total P and total N
        mass should not drift beyond floating-point tolerance."""
        state = _typical_state()
        params = ASM2dParams()
        dc = reactions_asm2d(state, params)

        p = params.asm1
        # Total P: S_PO4 + P bound in PAO biomass + poly-P store.
        d_total_p = dc[13] + params.i_p_bm * (dc[4] + dc[5] + dc[14]) + params.i_p_xi * dc[6] + dc[16]
        assert abs(float(d_total_p)) < 1e-4

        # Total N: S_NO + S_NH + S_ND + X_ND + N bound in biomass.
        d_total_n = dc[8] + dc[9] + dc[10] + dc[11] + p.i_xb * (dc[4] + dc[5]) + p.i_xp * dc[6] + params.i_n_bm * dc[14]
        assert abs(float(d_total_n)) < 1e-4


class TestPerfectClarifier:
    def test_effluent_has_no_particulates(self) -> None:
        state = _typical_state()
        effluent, _ = perfect_clarifier(state, jnp.array(600.0), jnp.array(500.0), jnp.array(10.0))
        assert float(effluent[14]) == 0.0  # X_PAO
        assert float(effluent[2]) == 0.0  # X_I
        assert float(effluent[13]) == float(state[13])  # S_PO4 passes through

    def test_return_sludge_concentrates_solids(self) -> None:
        state = _typical_state()
        _, return_sludge = perfect_clarifier(state, jnp.array(600.0), jnp.array(500.0), jnp.array(10.0))
        assert float(return_sludge[14]) > float(state[14])
        assert float(return_sludge[13]) == float(state[13])

    def test_shape(self) -> None:
        state = _typical_state()
        effluent, return_sludge = perfect_clarifier(state, jnp.array(600.0), jnp.array(500.0), jnp.array(10.0))
        assert effluent.shape == (N_COMPONENTS_ASM2D,)
        assert return_sludge.shape == (N_COMPONENTS_ASM2D,)
