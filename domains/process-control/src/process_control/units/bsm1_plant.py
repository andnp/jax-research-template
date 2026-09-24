from __future__ import annotations

import jax
import jax.numpy as jnp

from process_control.units.asm1 import ASM1Params, ASM1State, mix_streams
from process_control.units.asm1 import step as asm1_step


def perfect_clarifier(
    r5: ASM1State,
    q_aerobic: jax.Array,
    q_rs: jax.Array,
    q_w: jax.Array,
) -> tuple[ASM1State, ASM1State]:
    concentration_factor = q_aerobic / jnp.maximum(q_rs + q_w, 1.0)
    effluent = ASM1State(
        s_i=r5.s_i, s_s=r5.s_s, x_i=jnp.array(0.0), x_s=jnp.array(0.0),
        x_bh=jnp.array(0.0), x_ba=jnp.array(0.0), x_p=jnp.array(0.0),
        s_o=r5.s_o, s_no=r5.s_no, s_nh=r5.s_nh, s_nd=r5.s_nd,
        x_nd=jnp.array(0.0), s_alk=r5.s_alk,
    )
    return_sludge = ASM1State(
        s_i=r5.s_i, s_s=r5.s_s, x_i=r5.x_i * concentration_factor,
        x_s=r5.x_s * concentration_factor, x_bh=r5.x_bh * concentration_factor,
        x_ba=r5.x_ba * concentration_factor, x_p=r5.x_p * concentration_factor,
        s_o=r5.s_o, s_no=r5.s_no, s_nh=r5.s_nh, s_nd=r5.s_nd,
        x_nd=r5.x_nd * concentration_factor, s_alk=r5.s_alk,
    )
    return effluent, return_sludge


def reactor_train(
    reactors: tuple[ASM1State, ASM1State, ASM1State, ASM1State, ASM1State],
    influent: ASM1State,
    q_in: jax.Array,
    return_sludge: ASM1State,
    q_rs: jax.Array,
    q_a: jax.Array,
    kla_34: jax.Array,
    kla_5: jax.Array,
    params: tuple[ASM1Params, ASM1Params, ASM1Params, ASM1Params, ASM1Params],
    dt: jax.Array,
) -> tuple[ASM1State, ASM1State, ASM1State, ASM1State, ASM1State]:
    q_total = q_in + q_rs + q_a
    inlet_rs, q_after_1 = mix_streams(influent, q_in, return_sludge, q_rs)
    inlet_r1, _ = mix_streams(inlet_rs, q_after_1, reactors[4], q_a)
    r1 = asm1_step(reactors[0], inlet_r1, q_total, jnp.array(0.0), params[0], dt)
    r2 = asm1_step(reactors[1], r1, q_total, jnp.array(0.0), params[1], dt)
    r3 = asm1_step(reactors[2], r2, q_total, kla_34, params[2], dt)
    r4 = asm1_step(reactors[3], r3, q_total, kla_34, params[3], dt)
    r5 = asm1_step(reactors[4], r4, q_total, kla_5, params[4], dt)
    return r1, r2, r3, r4, r5
