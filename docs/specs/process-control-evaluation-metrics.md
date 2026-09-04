# Process-Control Evaluation Metrics

The process-control package provides numerical trajectory metrics for evaluation
code. These functions operate on named arrays supplied by an environment or
evaluation harness; they do not define benchmark-specific sensor indices,
setpoints, reward policies, or safety thresholds.

All trajectory arrays use time on axis zero. Remaining axes are preserved, so a
`(time, seed)` input produces one result per seed. Integrals multiply each sample
sum by the declared control interval `dt`. Callers must provide signals in
consistent physical units and must record the units with their result artifact.

Tracking metrics accept an explicit signed error, conventionally
`measurement - setpoint`. They report IAE, ISE, maximum positive error, whether
the trace settles, and the first index from which every later sample remains
within tolerance. A trace that never settles reports `settled=False` and its
horizon length as the index.

Action metrics distinguish the requested command from the realized command.
Movement is computed from realized actions. Saturation uses explicit output
bounds. Rate limiting compares requested and realized actions and may also use
explicit asymmetric rate bounds. The first movement sample is zero unless an
initial action is supplied.

Constraint, safety, and fallback metrics consume explicit boolean masks or named
signals. Missing optional signals produce no metric; they are never interpreted
as zero violations. Economic metrics integrate an explicitly supplied cost
signal and make no assumptions about reward composition.

Controller APIs remain separate from these metrics. A reusable controller accepts
named measurement and setpoint values and returns typed state plus diagnostics;
observation layouts and benchmark-specific signal mapping remain at the project
boundary.
