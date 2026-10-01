# Simple RL Batch Execution

> Status: IMPLEMENTED in core (2026-10-01); projects migrate next. This
> supersedes the `*Hypers` split in `agent-param-split.md` and the trainer
> contract formerly described in `research-cli.md` and
> `parallel-experiment-runner.md`.

## Summary

Replace core's "trainer" layer with one field marker, one JAX function per
experiment and explicit metric declarations, the way rl-jax and
rl-control-template do it.

- A config field that may be vmapped says so: `traced(default)`.
- An experiment hands the runner a base config, named alternatives for parts
  of it, and a JAX `train(config, key)`. The runner groups runs, sets
  parameters by dotted path, and runs `jit(vmap(train))`.
- Anything emits named metrics. The experiment declares which ones to record,
  and how often: `add_metric("loop/reward", ema=0.999, every=500)`. Undeclared
  metrics are dropped on device. Core names, selects and derives no metric,
  and assumes no problem class.

## The whole API

This is everything an experiment author touches. Anything not listed is
private.

```python
from research_runner.config import traced
from research_runner.types import ExperimentSpec
from rl_agents.experiment import RLConfig, rl_train

experiment.add_parameter("env", ENV_NAMES)                     # a named choice
experiment.add_parameter("agent.LEARNING_RATE", [1e-3, 3e-4])  # a dotted config path
experiment.add_parameter("seed", range(30))
experiment.add_metric("loop/reward", ema=0.999, every=500)      # record, smoothed and thinned
experiment.add_metric("critic_loss", every=100)

ExperimentSpec(
    experiment,
    config=RLConfig(agent=PPOConfig(), env=GymnaxConfig("CartPole-v1")),
    choices={"env": {name: GymnaxConfig(name) for name in ENV_NAMES}},
    train=rl_train,
    results_root=RESULTS_ROOT,
)
```

| Name | What it is |
| --- | --- |
| `traced(default)` | marks a config field as vmappable |
| `add_parameter(path, values)` | sweeps a dotted config path, a named choice, or `seed` |
| `add_metric(name, *, every=1, ema=None, kind="float")` | records one emitted metric |
| `ExperimentSpec(experiment, *, config, choices={}, train, results_root, max_runs_per_batch=None)` | the experiment the runner executes |
| `RLConfig(agent, env, gamma=traced(0.99))` | the default RL config |
| `rl_train(config, key) -> (state, metrics)` | the default RL train function |
| `AgentConfig.build()`, `EnvConfig.build()` | each config builds its agent or environment |

That is seven names. `add_parameter` and `add_metric` already exist; their
behaviour changes as described below.

## Problem

Core needs about 500 lines to answer one question per field: "may this be
vmapped?" rl-jax answers it with a marker on the field.

Today a batch of runs goes through five layers:

1. `research_runner.BatchTrainer`: a protocol with `parameter_schema` and
   `__call__(ExecutionContext) -> ExecutionResult`.
2. `ParameterSchema` and `BoundBatch`: the set of vmappable names, and the
   batch's values checked against it.
3. `rl_agents.batch_trainer.GenericBatchTrainer` (361 lines): builds the schema
   and the `Config` for one algorithm.
4. `rl_agents.batch_registry.AlgorithmBatchTrainer` (133 lines): picks a
   trainer from the `algorithm` parameter.
5. `rl_components.batch.run_batch` and `rl_components.loop.run`: the
   `jit(vmap(...))` and the scan loop, which are the only parts doing RL.

The schema is not written anywhere. `_algorithm_metadata` rebuilds it from
three sources:

- the lower-cased field names of the agent's `*Hypers` struct;
- plus `INIT_SWEPT_FIELDS` on the agent class, plus `gamma` added by hand;
- minus a `_NEVER_DYNAMIC` list (`actor_percentile`, `uniform_weight`,
  `total_timesteps`).

Int versus float comes from comparing type hints across `Hypers` and `Config`.
`_config_kwargs` then turns flat lower-case parameter names back into
upper-case `Config` fields, with special cases for `env_name`, explicit `None`
and `Literal` values.

Metrics are hard-coded in core, in several places:

- each agent's `REPORTED_METRICS` tuple picks which of its own metrics survive;
- `project_standard_metrics` always keeps `reward`, and PPO's
  `project_policy_metrics` swaps it for a derived episodic curve,
  `returned_episode_returns`;
- one global `metric_subsample_factor` applies to every metric alike;
- `experiment.add_metric` says it whitelists metrics, but the runner writes
  every curve it is given.

Five small `Protocol` classes exist only to find `final_policy_params` inside
differently shaped agent states, so that two projects can evaluate the policy
on the host afterwards.

The cost shows up in four places:

- **Adding a vmappable hyperparameter** means editing a `Hypers` struct, the
  `Config`, and sometimes `INIT_SWEPT_FIELDS` or `_NEVER_DYNAMIC`. Forgetting
  one silently makes the field static.
- **Recording a metric** depends on which agent produced it and on core's
  assumptions about episodes, which are wrong for continual problems such as
  process control.
- **Reading an experiment** means knowing what a trainer, a spec, a schema, a
  bound batch and a projector are. None of these are RL words.
- **Every agent says the same thing twice**: a `Config` dataclass and a
  `Hypers` struct that mirror each other.

## Goals and non-goals

The test for every choice below: could a new student read one experiment file
and one agent file and know exactly what runs and what is recorded?

Goals:

- **One place says a field is vmappable**: `traced(default)` on the config
  field.
- **One place says what is recorded**: the experiment's `add_metric` lines.
  Core hard-codes no metric name, selection or derivation, and assumes no
  problem class (continual, episodic, prediction, control).
- **One config type per agent.** The `*Hypers` mirror struct is deleted.
- **A seven-name public API**, listed above. Every other helper is private.
- **Same speed or better**: runs that differ only in traced values share one
  compiled `jit(vmap(...))`, and undeclared metrics never leave the device.
- **No parallel paths**: the old layer and every project adapter built on it
  are removed in the same migration.

Non-goals:

- Changing the plan, execute and record lifecycle of `research-runner`
  (ADR-007), the SQLite schema's purpose, Slurm submission, or result storage.
- Changing any agent's learning maths or `rl_components.loop.run`'s semantics.
- Mixed algorithms inside one compiled batch. Different agents are different
  static configs and always were.
- Saving final agent state. No consumer reads it once evaluation moves into
  `train`; a checkpoint API is added when one does.

## Design

```text
Today                                     Proposed
-----                                     --------
research_runner                           research_runner
  asks the trainer for parameter_schema     sets dotted paths on the base config,
        |                                   static keys = params minus traced fields
AlgorithmBatchTrainer                             |
  intersects schemas, dispatches          jit(vmap(train))
        |                                   declared metrics sampled on device
GenericBatchTrainer                               |
  rebuilds the schema from Hypers + 2 lists train
        |                                   the project's JAX function
_config_kwargs                                    |
  lower-case names to UPPER Config fields  loop.run
        |                                   the RL interaction loop, unchanged
run_batch
  jit(vmap(train_one)) over dynamic axes
        |
projectors, REPORTED_METRICS
  core picks and derives metrics
        |
loop.run
  the RL interaction loop
```

Today, five layers sit between the runner and the loop, and core decides which
metrics survive. Afterwards the runner calls the project's `train` directly.
It reads vmappability off the config and recording off the experiment.

### 1. `traced()`: the only place vmappability is declared

```python
@chex_struct(frozen=True, kw_only=True)
class DQNConfig:
    LEARNING_RATE: float = traced(1e-3)
    TRAIN_FREQUENCY: int = traced(1)
    EPSILON_END: float = traced(0.05)
    BUFFER_SIZE: int = 10_000        # sets array shapes: static
    TOTAL_TIMESTEPS: int = 100_000   # scan length: static
    NETWORK_PRESET: Literal["mlp", "nature_cnn"] = "mlp"

    def build(self) -> DQNAgent:
        return DQNAgent(self)
```

The rule for marking a field is one sentence: **trace it if the agent only
uses it in JAX arithmetic; leave it static if Python reads it** (a shape, a
scan length, a branch, an `int()` cast, a validity check). This replaces
`*Hypers`, `INIT_SWEPT_FIELDS` and `_NEVER_DYNAMIC`.

A traced value's dtype comes from its default's Python type: `int` stacks as
int32, `float` as float32, `bool` as bool. That keeps today's `Hypers` dtypes.

`traced` lives in `research_runner.config` (about 80 lines, no JAX), carried
over from `simple-experiments/hypers.py`. Its walking, splitting and merging
helpers stay private to the runner.

### 2. Parameters set a base config

The spec's `config` is an ordinary config object: the experiment's defaults.
Each parameter name is either:

- a **dotted path** to a field, such as `agent.LEARNING_RATE` or `gamma`, whose
  value replaces that field; or
- a **choice**, a path listed in `choices`, such as `env`, whose value names
  which object goes there.

```python
choices={
    "agent": {"dqn": DQNConfig(), "qrc": QRCConfig(BETA=1.0)},
    "env": {name: GymnaxConfig(name) for name in ENV_NAMES},
}
```

Choices are applied before paths, so `agent.LEARNING_RATE` sets the field on
whichever agent `agent` picked. `seed` is the only reserved name. Values stay
flat JSON scalars, so storage, hashing, `--only` filters and
`research_analysis` keep working unchanged.

Errors name what is wrong at sync time, before anything runs:

- `agent.LEARNING_RTE`: "QRCConfig has no field LEARNING_RTE; fields are ...";
- `env="Acrobat-v1"`: "'Acrobat-v1' is not a choice for env; choices are ...";
- a value of the wrong type for its field: "agent.TRAIN_FREQUENCY expects int, got 2.5".

This replaces `rl_config`, the `AGENTS`/`ENVS` registries,
`make_algorithm_batch_trainer`, `make_env_factory` and `_config_kwargs`. A
project that adds an agent writes it into its own `choices` dict.

### 3. Metrics: anything emits, the experiment declares

This follows rl-control-template's `Collector(config={...}, default=Ignore())`.

**Emitting.** `train` returns a dict of named arrays. A per-step metric has a
leading time axis; a per-run value is 0-d. Names are whatever the emitter
chose: `loop/reward`, `env/chlorine_residual`, `critic_loss`,
`reward_last100`. Core never renames, selects or derives one.

**Declaring.** The experiment lists every metric it records. Anything not
declared is dropped:

```python
experiment.add_metric("loop/reward", ema=0.999, every=500)  # smoothed, then one in 500
experiment.add_metric("critic_loss", every=100)             # one in 100
experiment.add_metric("env/chlorine_residual")              # every step
experiment.add_metric("reward_last100")                     # a per-run scalar
experiment.add_metric("passed", kind="bool")
```

Two keywords cover every current use. `ema` smooths with an exponential moving
average. `every` keeps one value in every `n`. Smoothing always runs before
thinning, which is the only order that makes sense. A 0-d value takes neither.

**Conditional metrics.** Inside `experiment.when(...)` a metric is recorded only
by runs whose parameters match those conditions, as for `add_parameter`:

```python
with experiment.when(agent="sac"):
    experiment.add_metric("alpha", every=500)     # SAC runs only
with experiment.when(env="chlorine"):
    experiment.add_metric("env/chlorine_residual")
```

The same name may be declared under different conditions, but not twice under
the same ones. Conditions must be on parameters that are constant within a
batch: a declaration that matches only some runs of a batch, or that names a
parameter no run declares, fails with an error naming the metric.

**Where it happens.** The runner applies the declarations inside the
`jit(vmap(...))`. Undeclared metrics and the final agent state are never
outputs, so XLA can skip the work that only they need, and only thinned curves
reach the host.

**Checks.** A declared name that `train` does not emit fails the first batch
with an error naming both the missing key and the emitted keys. `every` or
`ema` on a 0-d value fails the same way.

**Identity.** `every` and `ema` are recording choices, not part of the study:
changing them does not re-plan runs, and neither do a metric's conditions. The
execution's artifact metadata records the declarations applied to each batch and the
values that produced each stored curve.

This replaces `REPORTED_METRICS`, the metric projectors, the PPO override and
the global `metric_subsample_factor`.

### 4. The spec and the runner

```python
@dataclass(frozen=True)
class ExperimentSpec[C]:
    experiment: Experiment
    config: C                                               # the base config
    train: Callable[[C, Key], tuple[object, Mapping[str, Array]]]
    results_root: Path
    choices: Mapping[str, Mapping[str, object]] = {}
    max_runs_per_batch: int | None = None
```

`train(config, key)` returns `(state, metrics)`, the same shape `loop.run`
returns, so `train=rl_train` needs no wrapper. The runner records the metrics
and discards the state inside the jit.

The runner does three things with a spec:

- **At sync**, it builds each run's config from its parameters. That reports
  any error above, and gives each run's static keys: its parameter names minus
  the traced fields it set. The keys go into the per-run `vmap_zone`, so
  grouping works as it does today.
- **At execute**, it runs one batch:

  ```python
  def run_batch(spec, configs, seeds):
      statics, values = zip(*map(split_traced, configs))   # equal statics: guaranteed by grouping
      def one(v, key):
          _, metrics = spec.train(merge_traced(statics[0], v), key)
          return {m.name: thin(m, require(metrics, m.name)) for m in spec.experiment.metrics}
      return jax.jit(jax.vmap(one))(stack(values), keys_for(seeds))
  ```

  It still rejects an empty batch, non-integer seeds and a config count that
  does not match the seed count.
- **At record**, each array's shape says what it is: `[runs, T]` is a curve and
  `[runs]` is a per-run scalar. The runner is an emitter too: it emits each
  batch's `runner/compile_seconds` and `runner/execute_seconds`, which are
  recorded as execution scalars only when the experiment declares them, like
  any other metric. No project times itself, and rates such as steps per
  second are derived in analysis.

There is no `Results` or `ExecutionResult` type, and no project `run`
function. `run_batch` is private to the runner.

### 5. RL defaults in `rl_agents.experiment`

About 20 lines:

```python
@chex_struct(frozen=True)
class RLConfig:
    agent: AgentConfig    # any agent config with .build()
    env: EnvConfig        # any env config with .build()
    gamma: float = traced(0.99)

def rl_train(config: RLConfig, key):
    env, env_params = config.env.build()
    return loop.run(config.agent.build(), env, key, steps=config.agent.TOTAL_TIMESTEPS,
                    gamma=config.gamma, env_params=env_params)
```

`rl_train` returns the loop's metrics untouched. Project extras wrap it in JAX
and emit more named values. Scores, evaluation rollouts and derived curves all
look like this:

```python
def train(config, key):
    final, m = rl_train(config, key)
    return final, {**m, "reward_last100": m["loop/reward"][-100:].mean(),
                   "eval/return": evaluate(final, config, key)}
```

`rl_components.metrics` stays a library of JAX helpers, such as
`episode_return_curve`, that a `train` may call. Nothing in core calls them.

### 6. Agents read their config directly

Today each agent turns its `Config` into a `*Hypers` struct of arrays in
`init`, stores it in state, and reads `state.hypers.F` in `step`. Because
`train` builds the agent inside the vmapped function, traced config fields are
already tracers there. So each agent:

- reads `self.config.F` directly;
- builds its optimizer with the traced learning rate, replacing the per-step
  `with_hyperparams` re-injection;
- drops the `hypers` leaf from its state and its `REPORTED_METRICS` and
  `INIT_SWEPT_FIELDS` attributes.

No learning maths changes. Agents still emit every metric they compute;
recording is the experiment's choice.

## Worked example

### An experiment file (learning-dynamics)

Before:

```python
_ENV_FACTORY = make_env_factory({name: gymnax_builder(name, normalized=True) for name in ENV_NAMES})

experiment.add_parameter("algorithm", ["ppo"])
experiment.add_parameter("learning_rate", list(learning_rates))   # dynamic? ask PPOHypers
experiment.add_parameter("env_name", list(ENV_NAMES))
experiment.add_parameter("seed", list(SEED_VALUES))
experiment.add_metric("returned_episode_returns")   # produced by a PPO branch in core

ExperimentSpec(experiment=experiment,
               train_fn=make_algorithm_batch_trainer(_ENV_FACTORY, ("ppo",)),
               max_runs_per_batch=32, metric_subsample_factor=100, ...)
```

After:

```python
def train(config, key):
    final, m = rl_train(config, key)
    return final, {**m, "returned_episode_returns": episode_return_curve(m)}

experiment.add_parameter("env", ENV_NAMES)
experiment.add_parameter("agent.LEARNING_RATE", learning_rates)  # traced on PPOConfig
experiment.add_parameter("seed", SEED_VALUES)
experiment.add_metric("returned_episode_returns", every=100)

ExperimentSpec(
    experiment,
    config=RLConfig(agent=PPOConfig(), env=GymnaxConfig(ENV_NAMES[0], NORMALIZED=True)),
    choices={"env": {name: GymnaxConfig(name, NORMALIZED=True) for name in ENV_NAMES}},
    train=train, max_runs_per_batch=32, ...)
```

Today the curve this spec records comes from a PPO-only branch in core. After
the change, the spec derives and declares it itself. Recording anything else,
such as `loop/reward` or `total_loss`, is one more `add_metric` line.

To find out whether the learning rate is vmapped, you open `PPOConfig` and look
for `traced(`. To find out what is recorded, you read the `add_metric` lines.

### An agent (DQN)

Before, spread across `dqn.py` and `replay_q_agent.py`:

```python
class DQNConfig:
    LEARNING_STARTS: int = 1_000
    ...

class ReplayQHypers:                     # 19 lines mirroring the config as jax.Array
    LEARNING_STARTS: jax.Array
    ...

def replay_q_hypers(config) -> ReplayQHypers:   # 12 lines of jnp.asarray(...)
    return ReplayQHypers(LEARNING_STARTS=jnp.asarray(config.LEARNING_STARTS, jnp.int32), ...)

class DQNAgent:
    REPORTED_METRICS = ("loss", "epsilon", "q_max")
    INIT_SWEPT_FIELDS = ()
    def init(self, key, spec): ... hypers=replay_q_hypers(self.config) ...
    def step(self, state, timestep, t): ... t >= state.algorithm_state.hypers.LEARNING_STARTS ...
```

After:

```python
class DQNConfig:
    LEARNING_STARTS: int = traced(1_000)
    ...
    def build(self) -> DQNAgent:
        return DQNAgent(self)

class DQNAgent:
    def step(self, state, timestep, t): ... t >= self.config.LEARNING_STARTS ...
```

### A project extra (actor-free-qrc)

Before, a decorator class around a dispatcher around a generic trainer:

```python
class ProductionScoreTrainer:
    def __init__(self, trainer: BatchTrainer): self._trainer = trainer
    @property
    def parameter_schema(self): return self._trainer.parameter_schema
    def __call__(self, context):
        result = self._trainer(context)
        threshold = CASES[context.get_static("env_name", str)].pass_threshold
        scores = [sum(c[-SCORE_WINDOW:]) / SCORE_WINDOW for c in result.metric_curves["reward"]]
        return replace(result, run_scalars={**result.run_scalars, "reward_last100": scores, ...})

train_fn = ProductionScoreTrainer(make_algorithm_batch_trainer(
    _ENV_FACTORY, (ALGORITHM,), extra_specs={ALGORITHM: _ALGORITHM_SPEC}))
```

After, a JAX function:

```python
def train(config, key):
    final, m = rl_train(config, key)
    score = m["loop/reward"][-SCORE_WINDOW:].mean()
    threshold = CASES[config.env.NAME].pass_threshold   # static: read in Python at trace time
    return final, {**m, "reward_last100": score, "passed": score >= threshold}

experiment.add_metric("reward_last100")
experiment.add_metric("passed", kind="bool")

ExperimentSpec(experiment, config=RLConfig(agent=ProductionBsuiteConfig(), env=CASE_ENVS[0]),
               choices={"env": CASE_ENVS_BY_NAME}, train=train, ...)
```

## What gets deleted

Core loses about 1,050 lines of source and gains about 200. Project glue
shrinks by several hundred more. Line counts are today's, from the files named.

| Removed | Where | Lines | Replaced by |
| --- | --- | --- | --- |
| `GenericBatchTrainer`, `AlgorithmSpec`, metric projectors, artifact extractor protocols, `_config_kwargs`, `_NEVER_DYNAMIC` | `rl_agents/batch_trainer.py` | 361 | `rl_agents.experiment` (~20) |
| `AlgorithmBatchTrainer`, `ALGORITHM_TRAINERS`, `make_algorithm_batch_trainer` | `rl_agents/batch_registry.py` | 133 | the spec's `choices` |
| `*Hypers` structs and `*_hypers()` converters (7 registered agents, plus Rainbow and DQN-Atari) | `rl_agents/*.py` | ~220 | `traced()` on existing config fields |
| `REPORTED_METRICS`, `INIT_SWEPT_FIELDS` | 7 agent classes | ~14 | `add_metric` declarations |
| `metric_subsample_factor` | `ExperimentSpec`, `research_runner/runner.py` | ~10 | `every=` per metric |
| `with_hyperparams` per-step re-injection | `rl_components/optim.py` | 10 | the optimizer built with the traced rate |
| `BatchTrainer`, `ParameterSchema`, `BoundBatch`, `ExecutionResult`, `get_parameter`, `_coerce`, `ExecutionContext.bind`/`axes`/`get_static` | `research_runner/types.py` | ~100 | `ExperimentSpec(config, choices, train)` |
| `_resolve_static_parameter_names` | `research_runner/runner.py` | 14 | traced fields of each run's config |
| `rl_components.batch.run_batch` | `rl_components/batch.py` | 32 | a private runner function (~25) |
| `ParameterAccessor`, `_MappingParameterAccessor`, `make_env_factory`, flat-mapping builders | `rl_envs/env_registry.py` | ~150 | env config dataclasses with `.build()` (~60) |
| Trainer tests | `tests/medium/test_rl_agents_batch_trainer.py` | 971 | ~250 lines testing configs, choices, metric recording and batching |

New code: `research_runner.config` (~80, including choices and errors),
metric thinning in `research_instrument` (~20), the private batch function
(~25), `rl_agents.experiment` (~20), env configs (~60).

Also rewritten, not deleted:

- the nine `*_vmap_zone.py` agent tests (879 lines), which build configs
  instead of `Hypers`;
- the "trainer" passages in `docs/workflow.md`, `docs/experiment-efficiency.md`
  and `docs/specs/research-cli.md`;
- the project template's `001_*.py.jinja`.

The vocabulary check after the migration:
`rg -i 'trainer|ParameterSchema|BoundBatch|Hypers\b|REPORTED_METRICS' core projects`
returns nothing.

## Migration

Seven projects move, easiest first. Each becomes a base config, maybe a
`choices` dict, maybe a JAX `train`, and explicit `add_metric` lines. None
needs a class.

| Project | Uses today | After | Glue owned today | Watch for |
| --- | --- | --- | --- | --- |
| learning-dynamics | algorithm dispatch | env `choices` + a `train` that emits the curve it declares | ~3 lines | its declared curve comes from core's PPO branch today; the spec now derives it. Two harness tests read the removed `HyperparamConfigRow.vmap_zone` |
| generative-replay | algorithm dispatch | env `choices` + `rl_train` | ~15 lines | `scenario` becomes a field on `ProcessControlConfig`; `test_harness_contract.py` reads the removed `HyperparamConfigRow.vmap_zone` |
| iterated-gtd | dispatch + `extra_specs` | agent `choices` including its GI-QRC configs | ~77 lines | `NUM_HEADS` and `ARCHITECTURE` stay static; drop the duplicate `sac_rc` name |
| actor-free-qrc | dispatch + decorator trainers | a `train` that emits scores | ~180 lines | `plateau_scalars` is a Python loop over curves; it moves to analysis or becomes JAX |
| process-control-baselines | dispatch wrapper, controller trainers, bare `run_batch` | `train` functions; evaluation rollouts run inside `train` after the loop and emit `eval/*` metrics | ~475 lines | its stored `telemetry` artifact becomes declared per-step metrics; controller rollouts are time-major, so emit time-first |
| reversible-rank-rl | 6 custom trainers, bespoke loops | an `MRPConfig` with traced `LR_PRIMAL`, `ETA`, `LAMBDA_DUAL`, plus one `train` | ~490 lines | `lr_dual = lr_primal * eta` moves into `train`; window scalars become JAX; the lane tuner reads runner-recorded batch timing |
| missingness-rl | fake `ExecutionContext` around the PPO trainer | a `train` with evaluation | ~800 lines | it writes `metrics.sqlite` itself today; its ragged per-episode curves become dense per-step metrics plus analysis; its four timing metrics become declared `runner/*` scalars, with `steps_per_second` and `wall_time_seconds` derived in its analysis |

The work runs in five phases. Each phase is a series of small commits on
`main` in the repository that owns it, planned with the `create-commit` skill
before any code is written.

1. **Core, additive.** Add `traced` and the private config helpers (paths,
   choices, errors) with their tests. Add metric thinning. Mark `traced()` on
   each agent config, one commit per agent, matching today's `Hypers` field
   sets. Keep `REWARD_SCALE` static, because PPO validates it in Python. Add
   the env config dataclasses and `rl_agents.experiment`.
2. **Core, the contract switch, in one bundled commit.** Give `ExperimentSpec`
   `config`, `choices` and `train` in place of `train_fn`, and make
   `add_metric` take `every` and `ema` and actually filter. Add the private
   batch function and record from its output. Delete `BatchTrainer`,
   `ParameterSchema`, `BoundBatch`, `ExecutionResult`,
   `metric_subsample_factor`, `batch_trainer.py`, `batch_registry.py` and
   `rl_components.batch`, and update the template and docs. The change breaks
   the contract, so the tests move with it.
3. **Projects**, one repository at a time, in the table's order. Each
   project's commits leave its own suite green against the new core.
4. **Core, cleanup.** One commit per agent: read `self.config.F` directly,
   then delete its `Hypers` struct, converter, state leaf,
   `REPORTED_METRICS` and `INIT_SWEPT_FIELDS`. Delete `with_hyperparams`.
5. **Monorepo.** Bump the `core` and project submodule pointers together in
   one commit, so the superproject never pins a core that its projects cannot
   import.

Parameter names change (`learning_rate` becomes `agent.LEARNING_RATE`), which
changes every hyperparameter-config hash. Completed studies keep their
databases and stay readable through `research_analysis`. A migrated spec that
is run again will re-plan its runs, as `docs/experiment-efficiency.md` warns
for any re-identified study.

## Alternatives considered

| Option | Why not |
| --- | --- |
| Rename `BatchTrainer` to something friendlier and keep the rest | Cosmetic. Vmappability would still be rebuilt from three sources plus an exception list. |
| Keep `*Hypers` as the marker and drop `Config` instead | `Hypers` fields are all `jax.Array`, so static fields (shapes, scan lengths, `Literal` choices) would need a second home anyway. |
| Declare dynamic keys by hand in each spec file (`reversible-rank-rl` 005 does this) | Every experiment restates what the agent already knows, and the two drift apart. |
| A `config(params)` function plus core `AGENTS`/`ENVS` registries | Every project learns a builder, `partial`, and dict-merging into a registry. A base object plus `choices` says the same thing with data. |
| Sampler classes (`Identity`, `Subsample`, `Window`, `MovingAverage`, `Pipe`) | Composable in any order, but five classes for what two keywords cover. Add a class when a real use needs an order or a reduction the keywords cannot express. |
| Store every emitted metric and let analysis choose | Nothing is hard-coded, but a million-step continual run would copy every metric at full resolution off the device and into `metrics.sqlite`. |
| Sample on the host after the batch | Every metric's full curve for every run is materialised and copied off the device first. |
| A project `run` function and a `Results` type | They allow host-side Python after the batch, but every project keeps a function and a result type. `train` plus analysis covers the same ground. |
| A `keep=` hook to save final state | No consumer once evaluation moves into `train`. Add it when one appears. |
| Replace `experiment-definition` and `research-runner` with `simple-experiments` wholesale | It is the simplest end state, but it drops component versioning, the relational schema and `research_analysis`. That is a separate decision, and this design keeps it open because both use the same `traced()` model. |

## Decisions

Settled with Andy on 2026-09-29 and 2026-09-30. Three small items were left
for phase 1, and all three are now settled.

| Question | Decision |
| --- | --- |
| Parameter names | Dotted config paths (`agent.LEARNING_RATE`, `gamma`) plus named choices (`env`). Accept the re-hash: completed studies stay readable, and only specs that are re-run re-plan. |
| Config construction | A base config object plus `choices`. No core registries and no config-builder function. |
| Metrics | Core names, selects and derives no metric. Emitters name their own; the experiment declares each recorded metric; undeclared metrics are dropped. Modelled on rl-control-template's `Collector`. |
| Problem classes | None assumed. Continual, episodic and prediction problems use one path. PPO's `returned_episode_returns` branch is deleted. |
| Metric sampling | Two keywords, `ema` then `every`, applied inside `jit(vmap(...))`. Not part of study identity; the execution's artifact metadata records them. |
| Spec shape | `ExperimentSpec(experiment, config, choices, train)`. `train` returns `(state, metrics)` like `loop.run`; the runner discards the state on device. No `run` function, no `Results` type, no artifact hook. |
| Where `traced()` lives | `research_runner.config`. |
| Migration scope | All seven projects, in one migration. The old layer is deleted at the end. |
| Rainbow and DQN-Atari | They lose their `Hypers` structs in phase 4. Their `(config, runtime_config)` split stays. |
| Environment configs | Dataclasses with `.build()`. Wrappers such as normalisation and history frames become fields. `make_env_factory` is deleted. |

Settled in phase 1:

- [x] Confirm that `traced()` field metadata survives `chex_struct`. It does,
      including through `dataclasses.replace`
      (`tests/medium/test_research_runner_config_chex.py`), so agent configs
      stay `chex_struct` classes.
- [x] Keep `SAC.TARGET_ENTROPY` static, because whether it is `None` must
      agree across a batch. Trace `ALPHA` only.
- [x] Decide what `every=n` does when a run's length is not a multiple of `n`:
      keep the last partial step or drop it. `every=n` keeps steps 0, n, 2n,
      ... like ml_instrumentation's `Subsample`, so there is no partial block;
      `ema` matches its `MovingAverage`.
