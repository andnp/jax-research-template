"""Metadata and lookup helpers for process-control benchmarks."""

from __future__ import annotations

from dataclasses import dataclass


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


BENCHMARK_REGISTRY: tuple[BenchmarkEntry, ...] = (
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
    BenchmarkEntry(
        "bsm1",
        "process_control.benchmarks.bsm1",
        "BSM1BenchmarkConfig",
        "make_bsm1_benchmark",
        9,
        2,
        "BSM1 DO control (ASM1)",
    ),
    BenchmarkEntry(
        "bsm1_recycle",
        "process_control.benchmarks.bsm1_recycle",
        "BSM1RecycleConfig",
        "make_bsm1_recycle_benchmark",
        6,
        2,
        "BSM1 nitrate recycle",
    ),
    BenchmarkEntry(
        "bsm1_combined",
        "process_control.benchmarks.bsm1_combined",
        "BSM1CombinedConfig",
        "make_bsm1_combined_benchmark",
        11,
        4,
        "BSM1 combined control",
    ),
    BenchmarkEntry(
        "bsm1_lt",
        "process_control.benchmarks.bsm1_lt",
        "BSM1LTConfig",
        "make_bsm1_lt_benchmark",
        10,
        2,
        "BSM1-LT seasonal dynamics",
    ),
    BenchmarkEntry(
        "bsm1_takacs",
        "process_control.benchmarks.bsm1_takacs",
        "BSM1TakacsConfig",
        "make_bsm1_takacs_benchmark",
        12,
        3,
        "BSM1 + Takács settler",
    ),
    BenchmarkEntry(
        "h2s_scrubber",
        "process_control.benchmarks.h2s_scrubber",
        "H2SScrubberConfig",
        "make_h2s_scrubber_benchmark",
        12,
        3,
        "H₂S scrubber control",
    ),
    BenchmarkEntry(
        "sludge_blanket",
        "process_control.benchmarks.sludge_blanket",
        "SludgeBlanketConfig",
        "make_sludge_blanket_benchmark",
        4,
        1,
        "Sludge blanket control",
    ),
    BenchmarkEntry(
        "chem_p_dosing",
        "process_control.benchmarks.chem_p_dosing",
        "ChemPDosingConfig",
        "make_chem_p_dosing_benchmark",
        5,
        1,
        "Chemical P dosing",
    ),
    BenchmarkEntry(
        "primary_clarifier",
        "process_control.benchmarks.primary_clarifier",
        "PrimaryClarifierConfig",
        "make_primary_clarifier_benchmark",
        5,
        1,
        "Primary clarifier",
    ),
    BenchmarkEntry(
        "dewatering",
        "process_control.benchmarks.dewatering",
        "DewateringConfig",
        "make_dewatering_benchmark",
        5,
        2,
        "Sludge dewatering",
    ),
    BenchmarkEntry(
        "membrane_fouling",
        "process_control.benchmarks.membrane_fouling",
        "MembraneFoulingConfig",
        "make_membrane_fouling_benchmark",
        6,
        3,
        "Membrane fouling control",
    ),
    BenchmarkEntry(
        "anaerobic_digester",
        "process_control.benchmarks.anaerobic_digester",
        "AnaerobicDigesterConfig",
        "make_anaerobic_digester_benchmark",
        7,
        2,
        "Anaerobic digester",
    ),
    BenchmarkEntry(
        "reject_water",
        "process_control.benchmarks.reject_water",
        "RejectWaterConfig",
        "make_reject_water_benchmark",
        6,
        2,
        "Reject water management",
    ),
    BenchmarkEntry(
        "drinking_water_train",
        "process_control.benchmarks.drinking_water_train",
        "DrinkingWaterTrainConfig",
        "make_drinking_water_train_benchmark",
        8,
        3,
        "Drinking water treatment train",
    ),
    BenchmarkEntry(
        "bio_p",
        "process_control.benchmarks.bio_p",
        "BioPConfig",
        "make_bio_p_benchmark",
        7,
        2,
        "Biological P removal (ASM2d)",
    ),
    BenchmarkEntry(
        "combined_np",
        "process_control.benchmarks.combined_np",
        "CombinedNPConfig",
        "make_combined_np_benchmark",
        9,
        4,
        "Combined N+P control",
    ),
)


def get_benchmark_entry(name: str) -> BenchmarkEntry:
    for entry in BENCHMARK_REGISTRY:
        if entry.name == name:
            return entry
    available = [entry.name for entry in BENCHMARK_REGISTRY]
    raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
