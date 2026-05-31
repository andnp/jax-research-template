# Process Control Modeling Toolkit Appendices

These appendix pages support the main design doc. They are drill-down references for the shared architecture, not standalone design docs with separate scope or approval state.

- [Main architecture proposal](../2026-05-16-process-control-modeling-toolkit.md)
- [Streams, state, and ports](process-state-streams-and-ports.md)
- [Unit operations](unit-operations.md)
- [Sensors](sensors.md)
- [Actuators](actuators.md)
- [Chemistry and process blocks](chemistry-and-process-blocks.md)
- [Controllers and supervisory logic](controllers-and-supervisory-logic.md)
- [Disturbances and operating scenarios](disturbances-and-operating-scenarios.md)
- [Observations, benchmark contracts, and comparator design](observations-benchmarks-and-env-contracts.md)
- [Benchmark catalog](benchmark-catalog.md)
- [Uncertainty layers: disturbances, dynamics, and sensors](uncertainty-layers-disturbances-dynamics-and-sensors.md)
- [Software interface: shared transport, minimal protocols, and JAX-native execution](software-interface-shared-transport-and-jax-native-architecture.md)
- [H2S scrubber benchmark concept](h2s-scrubber-benchmark-concept.md)
- [Proposed implementation order](proposed-implementation-order.md)
- [Realism and operations extensions](realism-and-operations-extensions.md)
- [Validation, schema governance, assembly, and module certification](validation-schema-assembly-and-module-certification.md)
- [Wastewater RL environment roadmap](wastewater-rl-environment-roadmap.md)

Each appendix page follows the same pattern:

- what the module family is
- the contract it should satisfy
- the realistic principles to preserve
- the realism we are intentionally ignoring
- recommended examples and first-generation scope
