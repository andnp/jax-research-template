# Batched controller rollout contract

`rl_components.rollout` owns generic JAX mechanics for driving a stateful
controller across a batch of independent environment states. It is separate
from the agent-oriented training loop and does not define process-control
metric schemas.

The rollout is fixed-horizon. `reset_policy="ignore"` is the default: the
environment's post-step state remains in the carry even when a step reports a
boundary. This is the correct default for continuous plants. An episodic
caller must opt into `reset_policy="auto_reset"` and provide controller-state
reset behavior explicitly.

Every step preserves the environment boundary signals and `info` mapping in
the result. `info` keys and telemetry keys are static across the traced scan;
their values are JAX arrays with stable shapes and dtypes. Telemetry callbacks
may add named arrays but may not overwrite core metrics.

Fresh rollouts reserve a reset stream before the rollout stream. The existing
two-way mode splits environment and next-rollout keys; three-way mode splits
action, environment, and next-rollout keys. Continuations consume the carry's
existing RNG and previous action. `step_offset` changes callback indices, not
fresh-rollout masking or RNG progression.

Action shape normalization is explicit for scalar, singleton, and
multidimensional actions. Movement metrics retain all action dimensions.
Project-specific schemas such as `baseline` and `staged` remain adapters in
the consuming project.
