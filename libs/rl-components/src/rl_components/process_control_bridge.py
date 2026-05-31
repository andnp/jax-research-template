"""Process control benchmark adapter for the Gymnax-style GymEnv protocol.

Wraps a ``make_*_benchmark(config) -> (reset, step)`` function pair into
the tuple-returning GymEnv[ContinuousActionSpace] interface expected by
RL agents (TD3, SAC, etc.).

Usage::

    from process_control.benchmarks.bsm1 import BSM1BenchmarkConfig, make_bsm1_benchmark
    from rl_components.process_control_bridge import ProcessControlAdapter

    config = BSM1BenchmarkConfig()
    reset_fn, step_fn = make_bsm1_benchmark(config)
    env = ProcessControlAdapter(reset_fn, step_fn, obs_dim=9, action_dim=2, env_id="bsm1")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class _ObsSpace:
    shape: tuple[int, ...]
    dtype: jnp.dtype = jnp.float32


@dataclass(frozen=True)
class _ActionSpace:
    shape: tuple[int, ...]


class ProcessControlAdapter:
    """Adapts a process control benchmark to the GymEnv[ContinuousActionSpace] protocol.

    Process-control benchmarks use:
        reset(rng) -> (state, obs)
        step(state, action, rng) -> (state, obs, reward, done, info)

    GymEnv protocol expects:
        reset(key, params) -> (obs, state)
        step(key, state, action, params) -> (obs, state, reward, done, info)
    """

    def __init__(
        self,
        reset_fn: Callable,
        step_fn: Callable,
        obs_dim: int,
        action_dim: int,
        env_id: str = "process_control",
        max_steps: int = 10000,
        scalar_action: bool = False,
    ):
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._env_id = env_id
        self._max_steps = max_steps
        self._scalar_action = scalar_action
        self._obs_space = _ObsSpace(shape=(obs_dim,))
        self._action_space = _ActionSpace(shape=(action_dim,))

    def observation_space(self, params: object | None = None):
        return self._obs_space

    def action_space(self, params: object | None = None):
        return self._action_space

    def reset(self, key: jax.Array, params: object | None = None):
        state, obs = self._reset_fn(key)
        return obs, state

    def step(
        self,
        key: jax.Array,
        state: Any,
        action: jax.Array,
        params: object | None = None,
    ):
        if self._scalar_action:
            action = jnp.squeeze(action)
        new_state, obs, reward, done, info = self._step_fn(state, action, key)
        return obs, new_state, reward, done, info


# ── Registry of all process control benchmarks ──


@dataclass(frozen=True)
class BenchmarkEntry:
    name: str
    module: str
    config_cls: str
    make_fn: str
    obs_dim: int
    action_dim: int
    description: str
    scalar_action: bool = False


BENCHMARK_REGISTRY: list[BenchmarkEntry] = [
    BenchmarkEntry(
        "chlorine",
        "process_control.benchmarks.chlorine",
        "ChlorineBenchmarkConfig",
        "make_chlorine_benchmark",
        4,
        1,
        "Chlorine residual control",
        scalar_action=True,
    ),
    BenchmarkEntry(
        "chlorine_two_stage",
        "process_control.benchmarks.chlorine_two_stage",
        "ChlorineTwoStageBenchmarkConfig",
        "make_chlorine_two_stage_benchmark",
        4,
        1,
        "Two-stage chlorine",
        scalar_action=True,
    ),
    BenchmarkEntry(
        "ph_neutralization",
        "process_control.benchmarks.ph_neutralization",
        "PhNeutralizationBenchmarkConfig",
        "make_ph_neutralization_benchmark",
        4,
        1,
        "pH neutralisation",
        scalar_action=True,
    ),
    BenchmarkEntry(
        "equalization_tank",
        "process_control.benchmarks.equalization_tank",
        "EqualizationTankBenchmarkConfig",
        "make_equalization_tank_benchmark",
        4,
        1,
        "Equalization tank level control",
        scalar_action=True,
    ),
    BenchmarkEntry("bsm1", "process_control.benchmarks.bsm1", "BSM1BenchmarkConfig", "make_bsm1_benchmark", 9, 2, "BSM1 DO control (ASM1)"),
    BenchmarkEntry("bsm1_recycle", "process_control.benchmarks.bsm1_recycle", "BSM1RecycleConfig", "make_bsm1_recycle_benchmark", 6, 2, "BSM1 nitrate recycle"),
    BenchmarkEntry("bsm1_combined", "process_control.benchmarks.bsm1_combined", "BSM1CombinedConfig", "make_bsm1_combined_benchmark", 11, 4, "BSM1 combined control"),
    BenchmarkEntry("bsm1_lt", "process_control.benchmarks.bsm1_lt", "BSM1LTConfig", "make_bsm1_lt_benchmark", 10, 2, "BSM1-LT seasonal dynamics"),
    BenchmarkEntry("bsm1_takacs", "process_control.benchmarks.bsm1_takacs", "BSM1TakacsConfig", "make_bsm1_takacs_benchmark", 12, 3, "BSM1 + Takács settler"),
    BenchmarkEntry("h2s_scrubber", "process_control.benchmarks.h2s_scrubber", "H2SScrubberConfig", "make_h2s_scrubber_benchmark", 12, 3, "H₂S scrubber control"),
    BenchmarkEntry("sludge_blanket", "process_control.benchmarks.sludge_blanket", "SludgeBlanketConfig", "make_sludge_blanket_benchmark", 4, 1, "Sludge blanket control"),
    BenchmarkEntry("chem_p_dosing", "process_control.benchmarks.chem_p_dosing", "ChemPDosingConfig", "make_chem_p_dosing_benchmark", 5, 1, "Chemical P dosing"),
    BenchmarkEntry("primary_clarifier", "process_control.benchmarks.primary_clarifier", "PrimaryClarifierConfig", "make_primary_clarifier_benchmark", 5, 1, "Primary clarifier"),
    BenchmarkEntry("dewatering", "process_control.benchmarks.dewatering", "DewateringConfig", "make_dewatering_benchmark", 5, 2, "Sludge dewatering"),
    BenchmarkEntry("membrane_fouling", "process_control.benchmarks.membrane_fouling", "MembraneFoulingConfig", "make_membrane_fouling_benchmark", 6, 3, "Membrane fouling control"),
    BenchmarkEntry(
        "anaerobic_digester",
        "process_control.benchmarks.anaerobic_digester",
        "AnaerobicDigesterConfig",
        "make_anaerobic_digester_benchmark",
        7,
        2,
        "Anaerobic digester",
    ),
    BenchmarkEntry("reject_water", "process_control.benchmarks.reject_water", "RejectWaterConfig", "make_reject_water_benchmark", 6, 2, "Reject water management"),
    BenchmarkEntry(
        "drinking_water_train",
        "process_control.benchmarks.drinking_water_train",
        "DrinkingWaterTrainConfig",
        "make_drinking_water_train_benchmark",
        8,
        3,
        "Drinking water treatment train",
    ),
    BenchmarkEntry("bio_p", "process_control.benchmarks.bio_p", "BioPConfig", "make_bio_p_benchmark", 7, 2, "Biological P removal (ASM2d)"),
    BenchmarkEntry("combined_np", "process_control.benchmarks.combined_np", "CombinedNPConfig", "make_combined_np_benchmark", 9, 4, "Combined N+P control"),
]


def get_benchmark_entry(name: str):
    for entry in BENCHMARK_REGISTRY:
        if entry.name == name:
            return entry
    available = [e.name for e in BENCHMARK_REGISTRY]
    raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")


def make_adapter(name: str, **config_overrides):
    """Create a ProcessControlAdapter by benchmark name.

    Args:
        name: benchmark name (e.g. "bsm1", "chlorine", "membrane_fouling")
        **config_overrides: keyword overrides for the benchmark config

    Returns:
        ProcessControlAdapter wrapping the named benchmark.
    """
    import importlib

    entry = get_benchmark_entry(name)
    mod = importlib.import_module(entry.module)
    config_cls = getattr(mod, entry.config_cls)
    make_fn = getattr(mod, entry.make_fn)

    config = config_cls(**config_overrides) if config_overrides else config_cls()
    reset_fn, step_fn = make_fn(config)

    return ProcessControlAdapter(
        reset_fn=reset_fn,
        step_fn=step_fn,
        obs_dim=entry.obs_dim,
        action_dim=entry.action_dim,
        env_id=f"process_control:{entry.name}",
        scalar_action=entry.scalar_action,
    )
