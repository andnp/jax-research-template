import jax


def compute_demand(ammonia: jax.Array, organics: jax.Array, turbidity: jax.Array) -> jax.Array:
    """Chlorine (mg Cl2/L) demand pool from influent constituents (mg/L, mg/L, NTU).

    ammonia coefficient (7.6) is the breakpoint stoichiometric Cl2:NH3-N mass
    ratio (White's Handbook of Chlorination & Alternative Disinfectants, ch.
    4; EPA/WHO breakpoint guidance). organics (0.5) and turbidity (0.02)
    coefficients are reduced-order placeholders, not calibrated to a
    specific source water or literature value.
    """
    return ammonia * 7.6 + organics * 0.5 + turbidity * 0.02
