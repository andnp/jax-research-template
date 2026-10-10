"""Metadata and lookup helpers for process-control benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax

from process_control.actuators.dosing_system import DIRECT

type ActionBound = float | tuple[float, ...]
type ConfigBound = str | tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkEntry:
    name: str
    module: str
    config_cls: str
    make_fn: str
    action_dim: int
    description: str
    scalar_action: bool = False
    action_low: ActionBound | None = None
    action_high: ActionBound | None = None
    action_low_config: ConfigBound | None = None
    action_high_config: ConfigBound | None = None
    action_control_mode_config: str | None = None
    action_control_mode: int | None = None


BENCHMARK_REGISTRY: tuple[BenchmarkEntry, ...] = (
    BenchmarkEntry(
        "chlorine",
        "process_control.benchmarks.chlorine",
        "ChlorineBenchmarkConfig",
        "make_chlorine_benchmark",
        1,
        "Chlorine residual control",
        scalar_action=True,
        action_low_config="pump_min_dose",
        action_high_config="pump_max_dose",
        action_control_mode_config="control_mode",
        action_control_mode=DIRECT,
    ),
    BenchmarkEntry(
        "ph_neutralization",
        "process_control.benchmarks.ph_neutralization",
        "PhNeutralizationBenchmarkConfig",
        "make_ph_neutralization_benchmark",
        1,
        "pH neutralisation",
        scalar_action=True,
        action_low_config="pump_min_dose",
        action_high_config="pump_max_dose",
    ),
    BenchmarkEntry(
        "equalization_tank",
        "process_control.benchmarks.equalization_tank",
        "EqualizationTankBenchmarkConfig",
        "make_equalization_tank_benchmark",
        1,
        "Equalization tank level control",
        scalar_action=True,
        action_low_config="pump_min_flow",
        action_high_config="pump_max_flow",
    ),
    BenchmarkEntry(
        "bsm1",
        "process_control.benchmarks.bsm1",
        "BSM1BenchmarkConfig",
        "make_bsm1_benchmark",
        2,
        "BSM1 DO control (ASM1)",
        action_low_config=("kla_34_min", "kla_5_min"),
        action_high_config=("kla_34_max", "kla_5_max"),
    ),
    BenchmarkEntry(
        "bsm1_recycle",
        "process_control.benchmarks.bsm1_recycle",
        "BSM1RecycleConfig",
        "make_bsm1_recycle_benchmark",
        2,
        "BSM1 nitrate recycle",
        action_low_config=("q_a_ratio_min", "q_rs_ratio_min"),
        action_high_config=("q_a_ratio_max", "q_rs_ratio_max"),
    ),
    BenchmarkEntry(
        "bsm1_combined",
        "process_control.benchmarks.bsm1_combined",
        "BSM1CombinedConfig",
        "make_bsm1_combined_benchmark",
        4,
        "BSM1 combined control",
        action_low_config=("bsm1.kla_34_min", "bsm1.kla_5_min", "recycle.q_a_ratio_min", "recycle.q_rs_ratio_min"),
        action_high_config=("bsm1.kla_34_max", "bsm1.kla_5_max", "recycle.q_a_ratio_max", "recycle.q_rs_ratio_max"),
    ),
    BenchmarkEntry(
        "bsm1_lt",
        "process_control.benchmarks.bsm1_lt",
        "BSM1LTConfig",
        "make_bsm1_lt_benchmark",
        2,
        "BSM1-LT seasonal dynamics",
        action_low_config=("bsm1.kla_34_min", "bsm1.kla_5_min"),
        action_high_config=("bsm1.kla_34_max", "bsm1.kla_5_max"),
    ),
    BenchmarkEntry(
        "bsm1_takacs",
        "process_control.benchmarks.bsm1_takacs",
        "BSM1TakacsConfig",
        "make_bsm1_takacs_benchmark",
        3,
        "BSM1 + Takács settler",
        action_low_config=("bsm1.kla_34_min", "bsm1.kla_5_min", "q_w_min"),
        action_high_config=("bsm1.kla_34_max", "bsm1.kla_5_max", "q_w_max"),
    ),
    BenchmarkEntry(
        "h2s_scrubber",
        "process_control.benchmarks.h2s_scrubber",
        "H2SScrubberConfig",
        "make_h2s_scrubber_benchmark",
        3,
        "H₂S scrubber control",
        action_low_config=("ph_sp_min", "orp_sp_min", "makeup_flow_sp_min"),
        action_high_config=("ph_sp_max", "orp_sp_max", "makeup_flow_sp_max"),
    ),
    BenchmarkEntry(
        "sludge_blanket",
        "process_control.benchmarks.sludge_blanket",
        "SludgeBlanketConfig",
        "make_sludge_blanket_benchmark",
        1,
        "Sludge blanket control",
        action_low=0.0,
        action_high=1.0,
    ),
    BenchmarkEntry(
        "primary_clarifier",
        "process_control.benchmarks.primary_clarifier",
        "PrimaryClarifierConfig",
        "make_primary_clarifier_benchmark",
        1,
        "Primary clarifier",
        action_low=0.0,
        action_high=1.0,
    ),
    BenchmarkEntry(
        "dewatering",
        "process_control.benchmarks.dewatering",
        "DewateringConfig",
        "make_dewatering_benchmark",
        2,
        "Sludge dewatering",
        action_low=(0.0, 0.0),
        action_high=(1.0, 1.0),
    ),
    BenchmarkEntry(
        "membrane_fouling",
        "process_control.benchmarks.membrane_fouling",
        "MembraneFoulingConfig",
        "make_membrane_fouling_benchmark",
        3,
        "Membrane fouling control",
        action_low=(0.0, 0.0, 0.0),
        action_high=(1.0, 1.0, 1.0),
    ),
    BenchmarkEntry(
        "anaerobic_digester",
        "process_control.benchmarks.anaerobic_digester",
        "AnaerobicDigesterConfig",
        "make_anaerobic_digester_benchmark",
        2,
        "Anaerobic digester",
        action_low=(0.0, 0.0),
        action_high=(1.0, 1.0),
    ),
    BenchmarkEntry(
        "reject_water",
        "process_control.benchmarks.reject_water",
        "RejectWaterConfig",
        "make_reject_water_benchmark",
        1,
        "Reject water management",
        action_low=(0.0,),
        action_high=(1.0,),
    ),
    BenchmarkEntry(
        "drinking_water_train",
        "process_control.benchmarks.drinking_water_train",
        "DrinkingWaterTrainConfig",
        "make_drinking_water_train_benchmark",
        3,
        "Drinking water treatment train",
        action_low=(0.0, 0.0, 0.0),
        action_high=(1.0, 1.0, 1.0),
    ),
    BenchmarkEntry(
        "combined_np",
        "process_control.benchmarks.combined_np",
        "CombinedNPConfig",
        "make_combined_np_benchmark",
        4,
        "Combined N+P control",
        action_low=(0.0, 0.0, 0.0, 0.0),
        action_high=(1.0, 1.0, 1.0, 1.0),
    ),
)


def get_benchmark_entry(name: str) -> BenchmarkEntry:
    for entry in BENCHMARK_REGISTRY:
        if entry.name == name:
            return entry
    available = [entry.name for entry in BENCHMARK_REGISTRY]
    raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")


def get_observation_dim(reset_fn: Callable[[jax.Array], tuple[object, jax.Array]]) -> int:
    """Derive the flattened observation width from a benchmark reset function."""
    shape = jax.eval_shape(reset_fn, jax.random.key(0))[1]
    return shape.shape[0]


def get_action_bounds(
    entry: BenchmarkEntry,
    config: object,
) -> tuple[ActionBound | None, ActionBound | None]:
    if entry.action_control_mode is not None:
        mode_config = entry.action_control_mode_config
        if mode_config is not None and _config_value(config, mode_config) != entry.action_control_mode:
            return None, None
    low = entry.action_low
    if entry.action_low_config is not None:
        low = _configured_bound(config, entry.action_low_config)
    high = entry.action_high
    if entry.action_high_config is not None:
        high = _configured_bound(config, entry.action_high_config)
    return low, high


def _configured_bound(config: object, fields: ConfigBound) -> ActionBound:
    if isinstance(fields, str):
        return _config_float(config, fields)
    return tuple(_config_float(config, field) for field in fields)


def _config_value(config: object, field: str) -> object:
    value = config
    for part in field.split("."):
        value = object.__getattribute__(value, part)
    return value


def _config_float(config: object, field: str) -> float:
    value = _config_value(config, field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configured action bound {field!r} must be numeric, got {value!r}")
    return float(value)
