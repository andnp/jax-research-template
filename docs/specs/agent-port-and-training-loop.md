# Technical Specification: Agent Port and Shared Training Loop

## 1. Overview

Every agent in `rl-agents` owns a private training loop. All ten (`dqn`, `double_dqn`,
`dueling_dqn`, `dqn_atari`, `rainbow`, `sac`, `td3`, `greedy_ac`, `qrc`, `ppo`) repeat the
same skeleton inside `make_train`:

```
build network/optimizer/buffer -> env.reset -> _update_step(runner_state, t):
    act -> env.step -> buffer.add -> lax.cond(can_train, learn) -> lax.cond(t % target_freq, sync)
-> jax.lax.scan(_update_step, runner_state, jnp.arange(TOTAL_TIMESTEPS))
```

Two project agents (`projects/iterated-gtd/iterated_gtd/agent.py`,
`projects/gac-gradient-refinement/gac_gradient_refinement/agent.py`) copy the skeleton
again, so every new project re-implements the loop. A third project consumes library
`make_train` directly and must migrate with the agents it calls:
`process-control-baselines` imports `rl_agents.sac.make_train` and
`rl_agents.td3.make_train` from both `rl_comparison.py:156-158` and
`declarative.py:304,317`. The Rule of Three (principles §5) is satisfied roughly twelve
times over.

This specification defines an agent port (`AgentProtocol`), a step-level data contract
(`Timestep`), and one shared `run` loop, so that environment interaction and
episode-boundary accounting exist exactly once.

The correctness half of this document — §4, bootstrapping under termination and truncation
— is the substantive part. The refactor is a prerequisite for fixing it, not the other way
around: today the boundary logic is replicated in twelve places and is wrong in every one
of them.

## 2. Goals

- One implementation of environment interaction, episode-boundary accounting, and the
  `lax.scan` horizon.
- Agents that never reference an environment, and are therefore constructible and testable
  without one.
- Correct bootstrapping for continuation, termination, and truncation, decided once in the
  loop rather than per agent.
- Retire the legacy `rl_components.gym_env.GymEnv` tuple protocol in favour of the
  canonical `EnvProtocol` (see `specs/canonical-env-protocol.md`), which the loop becomes
  the sole consumer of.
- Preserve ADR 004: end-to-end JIT-able, "One Agent, One World", `vmap` over seeds outside
  the loop.

## 3. Non-Goals

- No vector-environment framework. The loop models one agent in one world; seed
  parallelism stays `jax.vmap(run, in_axes=(None, None, 0))` at the composition root.
- No change to `research-runner` or `research-instrument`. They already sit outside the
  inner loop and continue to consume scan-stacked metrics outside `jit`.
- No new algorithms. This is a re-seating of existing ones.
- No checkpointing design. The thesis `Checkpoint` pattern is out of scope here.

## 4. Bootstrap Semantics (Normative)

### 4.1 Two quantities, not one

For a transition `(s_t, a_t, r_t, s_{t+1})` two distinct quantities are required, and they
differ **exactly** on truncation:

| | bootstrap coefficient `d_t` (multiplies `V(s_{t+1})`) | trajectory break `episode_end_t` (stops backward recursions) |
|---|---|---|
| continuation | `gamma` | no |
| termination | **0** | yes |
| truncation | **`gamma`** | **yes** |

The rule is *bootstrap **at** a truncation, never **across** one*:

- **At**: the value estimate `V(s_{t+1})` is legitimate and must be used with coefficient
  `gamma`. The underlying MDP did not end; only the data collection did. Discarding this
  bootstrap biases every value near the time limit toward the truncated return.
- **Across**: no multi-step return — n-step, GAE, lambda-return, eligibility trace — may
  accumulate through the boundary. The buffer or rollout entry after a truncation belongs
  to a different episode. Accumulation must be cut at the boundary and bootstrapped there.

Collapsing the two into a single flag is wrong in one direction or the other:

- `done = terminated or truncated` used for both gets the break right and **kills the
  bootstrap at truncation**.
- `discount` used for both gets the bootstrap right and **accumulates across truncation**.

**Termination dominates.** `terminated` and `truncated` can both be true on one step: a
pole falls on exactly the step the cutoff is reached, or an adapter reports both (`brax`
does so today, §8.2). Precedence is normative:

```python
is_terminated = env_step.terminated
is_truncated  = (env_step.truncated | cutoff_reached) & ~is_terminated
```

Without this, a genuine terminal transition — where `V(s*)` is meaningless — would receive
`d = gamma` and bootstrap from it, biasing values upward.

### 4.2 Current behaviour is wrong in both roles

- `dqn.py:227` — `target = rewards + GAMMA * next_q_max * (1.0 - dones)`. `dones` is
  `terminated or truncated` from the gymnax bridge, so the bootstrap is killed at
  truncation. Same shape in `double_dqn`, `dueling_dqn`, `sac`, `td3`, `dqn_atari`.
- `ppo.py:199-201` — the same `not_done` drives both the TD residual and the GAE
  recursion. Both roles, one flag.
- `rainbow.py` — has a real in-agent n-step accumulator (`N_STEP = 3`,
  `_advance_n_step_accumulator`) that flushes on `done`, so the break is correct. The
  target is not: `rainbow.py:399` precomputes `bootstrap_discount =
  ADDITIONAL_DISCOUNT ** N_STEP` as a *compile-time scalar*, and
  `categorical_target_support(rewards, dones, support, discount: float)` takes a Python
  float and applies `(1.0 - dones[..., None])`. A variable-horizon truncated n-step
  transition is not representable at all (§4.5).
- `qrc.py:135` and `greedy_ac.py:334` already compute `gammas = gamma * (1.0 - dones)` —
  structurally the `discount` quantity of §4.1, merely derived from a fused `done`. For
  these two the change is a rename plus a corrected source.
- No agent reads `terminated` or `truncated` anywhere (`grep` over `rl-agents`: zero
  hits), even though `gymnax_bridge.py:81-82` puts both into `info`.
- `ReplayBufferState` stores a single `dones: bool_`, so no agent *could* carry the
  distinction.
- There is no episode-cutoff or time-limit mechanism anywhere in `libs/` (`grep` for
  `cutoff|TimeLimit|MAX_EPISODE`: zero hits). The thesis loop had one
  (`control/src/main.py:113`), but it treated the cut as an episode end for counting only,
  called `glue.start()`, and gave the agent nothing but `cleanup()`. The thesis is the
  right reference for loop *structure* and not for truncation bootstrapping.

### 4.3 The bootstrap state must survive the boundary

`d_t = gamma` at truncation is only correct if `V` is evaluated at the *true* next state.
This makes §4.1 and the reset ordering in §6 a single coupled decision, not two
independent ones.

Adopting `discount` **without** fixing reset ordering is strictly worse than today's code:
it would bootstrap from a post-reset observation with a non-zero coefficient, injecting an
unrelated episode's value into every truncated transition. Today's `(1 - done)`
accidentally masks that error. **These changes must land in the same commit.**

Three adapters currently destroy or mislabel the boundary state; see §8.2.

### 4.4 Time-limited versus time-unlimited tasks

There is one legitimate case where a time limit should *not* be bootstrapped: when the
horizon is part of the problem definition (a time-limited task), the optimal policy is
time-dependent, the limit is genuine termination, and remaining time belongs in the
observation. When the limit is a practical training device (a time-unlimited task),
partial-episode bootstrapping as in §4.1 is correct.

Which case applies is a property of the *task*, so the default belongs on the environment
adapter, with a per-run override for experiments that deliberately vary it:

```python
TruncationPolicy = Literal["bootstrap", "terminate"]
```

- `EnvSpec.truncation_policy: TruncationPolicy = "bootstrap"` — the task's own answer.
- `run(..., truncation_policy=None)` — `None` defers to the spec; an explicit value
  overrides it and is recorded in the experiment row.

`"bootstrap"` maps `truncated -> d = gamma, episode_end = True`; `"terminate"` maps
`truncated -> d = 0, episode_end = True`. This is the only knob in the loop that changes
learning semantics.

### 4.5 Consequences for each algorithm family

**One-step targets** (DQN family, SAC, TD3, QRC, GreedyAC). `target = r + d * max_a Q(s', a)`.
The stored `discount` replaces `dones` entirely; these agents get simpler, not more complex.

**GAE** (PPO). The recurrence is:

```python
delta = ts.reward + ts.discount * bootstrap_value - value           # coefficient
gae   = delta + ts.discount * GAE_LAMBDA * (1.0 - ts.episode_end) * gae  # break
```

`ts.discount * lam * (1 - episode_end)` equals the textbook `gamma * lam * (1 - done)` in
all three cases, because wherever `discount != gamma` the `(1 - episode_end)` factor is
already zero. The equivalence is conditional on `done` meaning exactly `episode_end` — i.e.
`terminated | truncated` after termination dominance — not on some independently sourced
flag. It holds under both `truncation_policy` values. PPO therefore needs no `gamma` of its
own.

**The rollout slot layout is normative, because the existing GAE code is indexed
differently and reusing it unchanged is a one-off misalignment.** `ppo.py:195-206` scans
backward carrying `(gae, next_value)` and reads `transition.value` for the *following*
step, i.e. it assumes each slot holds `(obs_t, a_t, r_t, done_t)` — reward and done
arriving *with* the action. This port delivers reward and discount with the *next*
observation (§6.1).

Slots are indexed by the iteration at which a transition **completes**, not the one that
opened it. This is forced by `lax.scan`: a scan body at iteration `i` can only emit values
it holds at iteration `i`, so a layout referring to `ts_{i+1}` is not implementable in a
single forward pass. Slot `i` therefore records the transition closed at iteration `i`:

| field | value |
|---|---|
| `obs`, `action`, `log_prob`, `value` | the `(s_{i-1}, a_{i-1}, log_prob, V(s_{i-1}))` the agent carries in its own state |
| `reward`, `discount`, `episode_end` | from the incoming `ts_i` |
| `bootstrap_value` | `V(ts_i.bootstrap_observation)`, computed at iteration `i` |
| `valid` | `step_index > 0` |

`step_index` counts iterations of the whole run, not of the current rollout, so
`valid` is false on exactly **one** iteration ever: global `t = 0`. PPO's first rollout
therefore holds `NUM_STEPS - 1` valid transitions and every rollout after it holds
`NUM_STEPS`. The mask is not a per-rollout boundary artefact, and the minibatch losses take
a masked mean — `jnp.sum(loss * valid) / jnp.maximum(jnp.sum(valid), 1.0)` — so the single
invalid slot costs nothing beyond the first update.

That the agent supplies `(s_{i-1}, a_{i-1})` from its own state is the same property §5
requires of every agent, so this costs no extra machinery.

`bootstrap_value` must be stored per slot, not recovered by carrying `next_value` backward
through the scan. Carrying it backward is what reintroduces cross-episode leakage: at a
boundary the following slot's `value` is `V(s_0_new)`, and with `d = gamma` on truncation
that value would enter the target. Computing the TD residual forward, where the true
`bootstrap_observation` is in hand, decouples advantage accumulation from value
propagation entirely.

**Multi-step returns** (Rainbow n-step, eligibility traces). The n-step target is

```
G_t^(n) = sum_{k<n} (prod_{j<k} d_{t+j}) r_{t+k} + (prod_{j<n} d_{t+j}) V(s_{t+n})
```

The product handles *termination* automatically — once `d = 0` the tail vanishes — but
**not** truncation, where `d = gamma != 0` and the sum would run into the next episode.
The accumulator must additionally cut at the first `episode_end` in its window.

**A boundary flushes the whole window at different horizons, so a boolean cannot encode
it.** With `n = 3` and a boundary `m` steps ahead, the item at offset `0` has horizon `m`,
the item at offset `1` has horizon `m - 1`, and so on. Each flushed transition therefore
carries a *different* cumulative discount. What must be materialised per transition is the
product itself:

```
D_t = prod_{j<m_t} d_{t+j},    m_t = min(n, steps_to_first_episode_end)
```

`D_t` alone is not sufficient: a shortened window also needs the bootstrap state at *its*
horizon, not the one at the end of a full-length window. The flushed transition is defined
completely as:

```
m_t        = min(n, steps from t to the first episode_end inclusive)
reward_t   = sum_{k<m_t} (prod_{j<k} d_{t+j}) r_{t+k}
discount_t = prod_{j<m_t} d_{t+j}
next_obs_t = the bootstrap_observation at t + m_t
```

`next_obs_t` must be the true boundary state on a truncation (§4.3). Getting `discount_t`
right while leaving `next_obs_t` pointing at the window's nominal end is a silent
truncation bug that the `D_t` change alone would not catch.

The replay entry stores `discount = D_t` as a `float32` array and **no** boolean. This
supersedes the earlier draft's "additionally store `last`", which cannot represent
variable horizons. Concretely for Rainbow: delete the compile-time
`bootstrap_discount = ADDITIONAL_DISCOUNT ** N_STEP` (`rainbow.py:399`), change
`categorical_target_support`'s `discount: float` to a `jax.Array` broadcastable over the
batch, and drop its `(1.0 - dones[..., None])` factor — `D_t` already carries it.

Crucially, all of this stays **inside the agent**. The loop supplies
`(reward, discount, episode_end)` per step and knows nothing about n-step.

## 5. Ports

Placed in `rl-components` alongside the existing `env_protocol`.

```python
# rl_components/timestep.py
@chex_struct(frozen=True)
class Timestep[ObservationT]:
    # --- completes the transition begun by the agent's previous action ---
    reward: jax.Array                    # reward for that action; 0.0 when no transition
    discount: jax.Array                  # bootstrap coefficient: 0 on termination, gamma otherwise
    bootstrap_observation: ObservationT  # the TRUE state reached; never a post-reset observation
    episode_end: jax.Array               # bool: that transition ended an episode (termination OR truncation)
    # --- begins the next action ---
    observation: ObservationT            # the state to act from; post-reset at a boundary
```

Every field has exactly one job, and the two of §4.1 are named rather than inferred.
`bootstrap_observation == observation` except on a boundary step. There is no
`transition_valid` field: iteration `0` is the only one that closes no transition, so
validity is exactly `step_index > 0`. Carrying a second representation of that would only
invite the two to diverge. The agent's own ordering
within `step` is normative: **complete the pending transition using
`bootstrap_observation`, then reset traces if `episode_end`, then select an action from
`observation`.**

A single `step` method under `lax.scan` cannot enforce statement order structurally — an
agent that writes `trace = jnp.where(ts.episode_end, 0.0, trace)` at the top of `step`
silently drops the last update of every episode, and nothing in the types objects. This
ordering is therefore guarded by test, not by construction (§10).

`terminated` / `truncated` are deliberately not duplicated onto `Timestep`. Given §4.4 they
are recoverable, and the loop reports the episode-end reason through metrics instead.

```python
# rl_components/agent_protocol.py
@chex_struct(frozen=True)
class AgentStep[AgentStateT, ActionT]:
    state: AgentStateT
    action: ActionT
    metrics: dict[str, jax.Array]

@runtime_checkable
class AgentProtocol[AgentStateT, ObservationT, ActionT](Protocol):
    def init(self, key: chex.PRNGKey, spec: EnvSpec) -> AgentStateT: ...
    def step(
        self, state: AgentStateT, timestep: Timestep[ObservationT], step_index: jax.Array
    ) -> AgentStep[AgentStateT, ActionT]: ...
```

Two methods. Properties that follow:

- **The agent never sees `env`** — only `EnvSpec` at init and `Timestep` at step. A unit
  test can drive `step` with a hand-built timestep and assert an exact target.
- **`(s, a)` stays in agent state.** The loop supplies `(r, d, s')`; the agent remembers
  what it did. This is RL-Glue's contract, and it is what makes replay insertion the
  agent's business rather than the loop's.
- **Learning lives inside `step`**, under `lax.cond(can_train, ...)` — which is what
  `dqn.py:255` already does.
- **`cleanup()` disappears.** Trace resets become
  `jnp.where(timestep.episode_end, 0.0, trace)`.
- **`step_index` counts transitions *closed* so far**, which is what replay contents and
  `LEARNING_STARTS` actually mean, so `t % TRAIN_FREQUENCY`, `LEARNING_STARTS`, and epsilon
  schedules key off collected experience rather than loop iterations. Note the asymmetry:
  iteration `0` closes nothing but opens the first transition, which is closed and recorded
  at iteration `1` — nothing is discarded at the start. The action taken on the *final*
  iteration opens a transition that is never closed, which is unavoidable without an extra
  environment step and is the one transition a run of `N` iterations does not record. Under the
  §6.2 design these coincide, but passing it explicitly keeps agents from re-deriving it
  and makes the guarantee checkable.

**`init` receives no observation, so the pending-transition slots must be zero-primed from
`EnvSpec`.** Because a transition now closes on the iteration *after* the action, every
agent's state carries `last_obs` and `last_action` (plus `log_prob` / `value` for PPO).
`init` must allocate these as zeros of `spec.observation_shape` / `spec.observation_dtype`
and `spec.action_shape` / `spec.action_dtype`. Leaving them unshaped is a JIT tracer shape
error on the first `step`; the zero values themselves are never used, because insertion is
guarded by `step_index > 0`.

### 5.1 JIT and pytree obligations

Two contracts that the type signatures do not express and that break at trace time if
violated:

- **The agent object is static, not a JAX value.** A Python object holding methods cannot
  be passed as a dynamic argument to a jitted `run`. The implementation and its
  configuration are closed over or marked static; only `AgentStateT`, a pytree, moves
  through the scan carry. The same applies to `env` and to `env_params`, which is static
  unless an experiment sweeps it.
- **`metrics` needs a fixed key schema.** Every scan iteration must return the same
  dictionary keys with the same leaf shapes and dtypes. An agent that emits a metric only
  on the iterations it learns, or that returns `{}` before `LEARNING_STARTS`, produces a
  pytree mismatch under `scan`. The schema is fixed at `init` and unavailable metrics are
  emitted as zero-valued placeholders.

### 5.2 Why RL-Glue's method set is not used verbatim

RL-Glue is `start() -> a`, `step(r, s) -> a`, `end(r)`. Under `jax.lax.scan` every
iteration must return an identically-shaped pytree, so `end` returning no action is not
expressible. The dm_env encoding — one `step`, with boundary information carried in the
data — is the JAX-compatible form of the same semantics. This is an encoding change, not a
semantic one.

## 6. The Loop

```python
def run(
    agent: AgentProtocol[AgentStateT, ObsT, ActT],
    env: EnvProtocol[ObsT, EnvStateT, ActT, ParamsT],
    key: chex.PRNGKey,
    *,
    steps: int,
    gamma: float,
    episode_cutoff: int = -1,
    truncation_policy: TruncationPolicy | None = None,
    env_params: ParamsT | None = None,
) -> tuple[LoopState, dict[str, jax.Array]]:
```

with:

```python
@chex_struct(frozen=True)
class LoopState[EnvStateT, AgentStateT, ObsT]:
    env_state: EnvStateT
    agent_state: AgentStateT
    timestep: Timestep[ObsT]     # the timestep the agent is handed next
    episode_length: jax.Array    # int32
    episode_return: jax.Array    # float32
    key: chex.PRNGKey
```

It owns, once: the initial reset; the `(terminated, truncated, cutoff, policy)` to
`(discount, episode_end)` mapping of §4 including termination dominance; `episode_cutoff`
as truncation; the boundary reset; episode return and length accumulation emitted as loop
metrics alongside the agent's; and the `lax.scan` over `steps`.

Before the scan, `run` primes `timestep` from `env.reset`: `observation = s_0`,
`bootstrap_observation = s_0` (unused), `reward = 0.0`, `discount = 0.0`,
`episode_end = False`. The scan body then hands the agent the carried timestep, applies the
returned action, and builds the next one:

```python
# the true state reached always closes the pending transition
bootstrap_observation = env_step.observation

# the post-reset state supplies the observation the agent acts from next
next_observation = jax.tree_util.tree_map(
    lambda reset_v, step_v: jnp.where(episode_end, reset_v, step_v),
    reset.observation, env_step.observation,
)
next_env_state = jax.tree_util.tree_map(
    lambda reset_v, step_v: jnp.where(episode_end, reset_v, step_v),
    reset.state, env_step.state,
)
next_episode_length = jnp.where(episode_end, 0, episode_length + 1)
```

`bootstrap_observation` is never routed through the reset select — that substitution is
exactly the bug of §4.3.

This imposes a contract on adapters that `EnvProtocol` does not currently state: **the
pytree returned by `reset` and the one returned by `step` must have identical structure,
leaf shapes, and dtypes.** `jnp.where` cannot merge states whose leaves differ in shape, or
dicts whose keys differ. An adapter with optional or shape-dependent state leaves fails at
trace time, or silently promotes a dtype.

Dtype discipline for the same reason: reset and step values are constructed at the
adapter-declared dtype, and loop counters use explicit dtypes
(`jnp.zeros_like`, `jnp.asarray(..., dtype=...)`) rather than Python literals such as `0`
or `0.0`, whose weak typing can shift the scan carry away from the declared `LoopState`.

Two remaining semantics the loop must fix rather than leave to the implementer:

- **Cutoff timing.** The cutoff fires when `episode_length + 1 >= episode_cutoff` — that
  is, evaluated on the step that would exceed it — and it is applied *after* the
  environment's own flags, so termination dominance (§4.1) resolves the collision.
- **Keys.** The loop splits one key per iteration into a step key and a reset key, and the
  reset key is consumed on every iteration whether or not a boundary occurred, so the
  stream does not depend on episode structure. Without this, two seeds with identical
  dynamics diverge as soon as their episode lengths differ.

`gamma` lives here and nowhere else. The per-agent `GAMMA` fields are deleted; two sources
of truth for the discount is precisely how this bug class recurs. This also makes
per-state discounting and continuing tasks expressible. The real supporting precedent is
`qrc.py:135` / `greedy_ac.py:334`, which already reduce to a per-transition `gammas` array.

**`ADDITIONAL_DISCOUNT` must be deleted, not preserved.** An earlier draft claimed it was
an independent factor multiplying a discount — true in DQN Zoo, where dm_env's discount is
a `{0, 1}` indicator and `additional_discount` supplies `gamma`. Here `Timestep.discount`
is `{0, gamma}`, so retaining `ADDITIONAL_DISCOUNT = 0.99` alongside `gamma = 0.99` would
double-discount to `0.9801` (`dqn_atari.py:235`, `rainbow.py:273`). Both configs drop the
field.

### 6.1 Indexing

Loop iteration `i` hands the agent a `Timestep` that simultaneously *closes* the transition
opened by `a_{i-1}` and *opens* the next by supplying `observation`. The completed
transition is `(obs_{i-1}, a_{i-1}, r_i, d_i, bootstrap_observation_i)`. Reward and discount
arrive with the observation that follows the action, as in dm_env.

Exactly one transition completes per iteration except iteration `0`, which opens the first
transition without closing one, so a run of `N` iterations yields `N - 1` transitions.
Step index and transition index therefore coincide, which is what keeps `TRAIN_FREQUENCY`,
`LEARNING_STARTS`, and epsilon schedules pinned to real experience — per seed as well as in
aggregate (§6.2).

### 6.2 Decision: loop-owned autoreset

When a boundary occurs, the loop — not the adapter, and not a wasted iteration — performs
the reset within the same scan step, and reports both observations:

- `bootstrap_observation = s_T`, the true final state, closing the previous transition with
  the correct `d`.
- `observation = s_0_new`, the post-reset state, from which the agent selects its next
  action.
- `episode_end = True`, breaking backward accumulation.

This preserves a strict 1:1 step-to-transition invariant. Two alternatives were considered
and rejected:

**Adapter-fused autoreset** (what gymnax does, and what the code inherits today):
`env.step` returns the post-reset observation on the terminating step, so `s_T` never
reaches the agent. Under `d = 0` this is merely lossy; under `d = gamma` at truncation it is
a correctness failure (§4.3). This is the status quo and is rejected outright.

**Non-fused reset**, spending one iteration on the boundary so the agent receives a
separate episode-start timestep — RL-Glue's literal `start()`. Correct, and conceptually
the cleanest, but it was rejected on three counts:

1. Every rollout and replay insert needs a validity mask, because boundary iterations
   produce no transition. (An earlier draft specified this mask as `~first`; it is `~last`.
   The off-by-one is illustrative of the cost.)
2. **Step-index dilation.** With 50-step episodes over 200k iterations, ~4k iterations are
   boundary bubbles. Any schedule keyed on the scan index — `t % TRAIN_FREQUENCY`,
   `t % TARGET_NETWORK_FREQUENCY`, `LEARNING_STARTS`, epsilon decay — drifts from real
   experience, and gradient steps fire on iterations that added nothing to replay.
3. Worse, under `jax.vmap(run)` across seeds the drift is *per-seed*: seeds hit boundaries
   at different iterations, so at a given scan index different seeds have collected
   different amounts of experience. Schedules stop being comparable across the seed
   dimension, which is the axis every result in this repo is averaged over.

The loop-owned reset costs one observation-sized field on `Timestep` and the ordering rule
in §5. That is cheaper than a mask in every agent plus a dilated step index.

### 6.3 Control flow

Implement the boundary reset with `jax.lax.cond`, not with masking or `jax.lax.select`.

Correctness does not rest on which branch executes. Both branches are traced and must be
traceable either way, and **`env.reset` must stay cheap and must be safe to call on every
step**. `cond` therefore changes only whether the reset is *evaluated*; it never changes
which value the loop selects.

Under `jax.vmap(run)` across seeds — the standard mode here — the boundary predicate is
per-seed and therefore batched, and `vmap` lawfully degrades the `cond` to a `select` that
evaluates both branches. That is exactly the masking behaviour, so the batched path loses
nothing.

On the unbatched path `cond` is load-bearing rather than an optimisation. The ALE bridge
(`python_env_bridge.py`) holds its emulator state in Python behind an ordered
`io_callback`, outside the returned pytree. A `select` fires that callback on *every* step:
the selection discards the returned value while the side effect stands, so the emulator is
reset behind JAX's back and the running episode is destroyed — besides paying a full ALE
reset per step. Under `cond` the callback runs only on the iterations whose result is
actually selected, so the Python-side mutable state stays coherent with the JAX-side
selection.

Verified in this JAX version: an `ordered=True` `io_callback` inside `lax.cond` compiles,
and the untaken branch's callback does not fire.

This narrows ADR 004 §3 rather than contradicting it. Masking remains the rule for value
selection, where both branches are pure and kernel fusion is the only consideration. `cond`
is for guarding an *effectful* call, where evaluating the untaken branch is a semantic
change and not merely a cost.

### 6.4 End of the scan horizon

Reaching `steps` mid-episode is neither termination nor truncation of the MDP; it is simply
where data stops. `episode_end` is false there and the loop does nothing special. PPO's
existing `last_val` bootstrap at a rollout boundary is internal to the agent and is
unaffected.

## 7. Hexagonal Placement

| Role | Code |
|---|---|
| Ports | `env_protocol.EnvProtocol`, `agent_protocol.AgentProtocol`, `Timestep`, `EnvSpec` |
| Driven adapters (world) | `gymnax_bridge`, `brax`, `mujoco_playground_bridge`, `atari_ale`, `python_env_bridge`, `frame_stack` (decorator) |
| Driven adapters (policy) | `rl_agents.dqn.DQNAgent`, `.ppo.PPOAgent`, ... |
| Application | `rl_components.loop.run` |
| Composition root | the project's `train_fn(ExecutionContext) -> ExecutionResult` |

`process_control_bridge` is deliberately absent: it fills no `EnvStep` and is not an
`EnvProtocol` adapter. `frame_stack` is a decorator over an adapter rather than an adapter
proper, but it is listed because it independently performs a boundary reset (§8.2).

## 8. Required Downstream Changes

### 8.1 Replay buffer schema

`ReplayBufferState.dones: bool_` becomes `discount: float32`, carrying `d` for one-step
agents and the cumulative `D_t` of §4.5 for n-step agents. No boolean is stored: a bool
cannot represent a variable truncated horizon, and for one-step targets `discount`
subsumes `dones` outright (`target = r + discount * max Q`).

`ReplayBuffer` is shared, not DQN's: eight agents import it and unpack its 5-tuple as
`(obs, actions, rewards, next_obs, dones)` — `dqn`, `double_dqn`, `dueling_dqn`, `sac`,
`td3`, `qrc`, `greedy_ac`, `dqn_atari`. Swapping the fifth element to a `float32` discount
under them would leave `(1.0 - dones)` evaluating to `1 - 0.99 = 0.01` — silently wrong
rather than loudly broken. (`rainbow` is not a consumer; it has its own PER buffer, so it
is untouched here and takes the §4.5 change later.)

**This lands before any port work** (§9 step 2). Each consumer computes
`discount = gamma * (1.0 - done)` at insertion and uses `r + discount * max Q` in its
target. At that point `done` is still the fused flag from the gymnax bridge, so the change
is semantics-preserving: it changes representation, not behaviour, and every agent compiles
and passes its existing tests. When the loop arrives later, the buffer already speaks the
right language.

**Semantics-preserving is not bitwise-preserving, and the distinction was measured.** Where
an agent's old target put the mask last — `gamma * next_q * (1.0 - done)`, as in `dqn`,
`double_dqn`, `dueling_dqn` and `dqn_atari` — folding the discount reassociates the float
multiply chain to `(gamma * (1.0 - done)) * next_q`. That is exact in scalar IEEE-754, but
XLA rounds the two forms differently: measured over 4096 float32 elements, 812 differ, by a
mean of 1.35 ULP (max 101 ULP where `r + product` cancels), a maximum relative difference of
6.3e-06, and **only where `done == 0`** — the terminal case is exactly zero in both forms.
Fixed-seed runs confirm the replay contents, environment state, RNG stream and step metrics
stay byte-identical while the network parameters diverge.

Where the old target already put the mask first — `sac`, `td3` — and where the change only
moves an existing expression across a function boundary — `qrc`, `greedy_ac`, and both
Gi-QRC critics — the transformation *is* bitwise-identical, and that was confirmed by
snapshot rather than assumed.

The consequence for §10: a bitwise gate is the right check for the storage flip, whose
baseline is the state *after* this preparation. It is the wrong check across the
preparation itself, where the guarantee available is algebraic equivalence plus a bounded
ULP difference. Results computed before this batch reproduce statistically, not bitwise.

**The change spans eighteen files across three repositories, and therefore cannot be one
commit.** An earlier draft specified "one self-contained commit"; that is not achievable.
`core` is a git submodule with its own history, `projects/iterated-gtd` is an independent
repository not registered in `.gitmodules`, and `projects/gac-gradient-refinement` lives in
the outer monorepo. No commit spans two repositories.

What must hold is an **invariant**, not a commit count: *no committed state in any
repository may leave a consumer reading a boolean field as a discount.* That is satisfied
by consumer-side preparation first — each target expression rewritten to consume a
per-transition discount while the stored field is still a bool, which is bitwise-preserving
— followed by one storage flip per repository, landing back to back.

| consumer | why |
|---|---|
| `core/libs/rl-components/src/rl_components/buffers.py` | owns the field |
| `core/libs/rl-agents/src/rl_agents/{dqn,double_dqn,dueling_dqn,sac,td3,qrc,greedy_ac,dqn_atari}.py` | unpack `dones` from `buffer.sample` |
| `core/tests/small/test_rl_components_buffer.py:38-41` | `test_dones_dtype_is_bool` asserts `state.dones.dtype == jnp.bool_` |
| `core/tests/medium/test_rl_components_buffer_jit.py:7` | exercises the buffer under `jit` |
| `core/tests/medium/test_rl_agents_qrc_gradient.py` | calls the QRC batch loss directly (7 call sites) |
| `core/tests/performance/test_rl_agents_dqn_atari_env_loop_bench.py:259-320` | builds a `ReplayBufferState` fixture and casts `sample` results |
| `projects/iterated-gtd/iterated_gtd/{discrete,continuous}_gi_qrc.py` | unpack `dones` from `buffer.sample` |
| `projects/iterated-gtd/tests/small/test_{discrete_gi,gi}_critic_targets.py` | call the Gi-QRC critic losses directly |
| `projects/gac-gradient-refinement/gac_gradient_refinement/agent.py:19` | imports `ReplayBufferState` |

The inclusion criterion is precise and narrower than it looks: **consumers of the shared
`ReplayBuffer` schema**, not all code that learns from a fused `done`. Those are different
sets, and conflating them would make this commit unbounded. Learning code that uses a fused
`done` through its own batch abstraction is deliberately *out* of scope here and is fixed
when its owning agent ports:

- `projects/iterated-gtd/iterated_gtd/comparison.py:274,295` — `gamma * (1.0 - batch.done)`
  over a local batch type, no `ReplayBuffer`.
- `projects/iterated-gtd/iterated_gtd/discrete_comparison.py:126,149` — same pattern in a
  tabular comparison path.

Both carry the §4.1 truncation defect and neither is touched by step 2. They move in step
11 with the `iterated-gtd` port. (`projects/missingness-rl/missingness_rl/experiment.py:282`
uses `(1.0 - done)` for episode-return accounting, not a bootstrap target, and is correct
as written.)

Two of the listed project files are the same agents scheduled for migration in §9 step 11.
They are touched twice: once here for the representation change, once later for the port.
That is not a parallel path — the first change is unconditional and permanent — but the
ordering should not be a surprise when step 11 arrives.

Because `iterated-gtd` is neither a submodule nor pinned by `core` in either direction, no
committed state anywhere records the consistent triple. That is a pre-existing property of
the layout rather than something this change introduces, but it means the invariant above
is the only available guarantee, and it holds per repository rather than globally.

Three details make this mechanical rather than delicate:

- `dones` is used **only** in the target expression in all eight agents. No actor loss, no
  entropy term, and no metric reads it, so there is no second call site to keep in sync.
- `qrc.py:143` and `greedy_ac.py:334` already compute `gammas = gamma * (1.0 - dones)`
  inside their losses and take `gamma` as a separate argument. For those two the change
  deletes the multiply and drops the `gamma` parameter, rather than renaming a field.
- `dqn_atari.py:273` reads `buffer_state.dones[indices]` directly instead of going through
  `sample()`, so the field rename touches that line as well.

The reviewed alternative — add a second buffer symbol for migrated agents and delete the
legacy one at the end — is rejected: it is a parallel implementation of the same
component, which repository policy prohibits by default, and it would have kept both live
across six commits. The single behaviour-preserving commit exceeds the usual ~20-line
budget, which is the accepted trade for a contract change that cannot be split without
leaving a broken intermediate state.

### 8.2 Environment adapter defects

`EnvStep` already carries separate `terminated` / `truncated`. Every adapter filling it
gets the boundary wrong in some way, so the port cannot be fed correctly today. The first
two are live defects fixable now, independent of this refactor and independent of each
other. The rest cannot land until the loop exists and are scheduled in §9 accordingly.

- **`brax.py:132`** — `terminated=jnp.asarray(next_state.done)`. The adapter maps *every*
  Brax `done` to termination while separately reporting `truncated` from
  `info['truncation']`, so a time limit reports `terminated=True` **and**
  `truncated=True`, and termination dominance (§4.1) then zeroes a bootstrap that should
  have survived. The intended fix is `terminated = done & ~truncation`, but the exact
  `done`/`truncation` contract is Brax version- and environment-dependent: confirm it
  against the configured Brax version before changing the mapping. What the file alone
  establishes is that the current mapping cannot distinguish the two cases. Note also that
  `done` and `info['truncation']` are `float32`, and `jnp` `&`/`~` reject float dtypes, so
  the fix needs explicit bool casts.
- **`mujoco_playground_bridge.py:111`** — hardcodes `truncated=jnp.asarray(False)`, so
  every time limit is reported as termination. Playground routes its episode limit through
  the same brax `EpisodeWrapper`, so `info['truncation']` is the signal to read. Two
  environments fuse a step-count limit into `done` with no flag at all
  (`manipulation/aloha/handover.py:209`,
  `manipulation/franka_emika_panda/pick_cartesian.py:370`); the adapter cannot distinguish
  those, and they are the §4.4 `truncation_policy` case rather than an adapter bug.

Three further defects are **not** fixable before the loop exists, and are scheduled in §9
step 3 rather than here:

- **`brax.py:38`** — `BraxConfig.auto_reset` defaults `True`, so Brax's `AutoResetWrapper`
  replaces the returned observation with `info['first_obs']` on exactly the step that
  becomes `done`. The true boundary state never reaches the port. Fixing the
  `terminated`/`truncated` mapping makes truncation bootstrapping *expressible*; this makes
  it *correct*. Until both land, a truncated transition with `d = gamma` would bootstrap
  from a post-reset observation — the §4.3 failure, and strictly worse than today.
- **`python_env_bridge.py:92-93`** — fused autoreset inside the `io_callback`: on
  `terminated or truncated` it calls `self._env.reset()` and returns the post-reset
  observation. The adapter must stop resetting and let the loop own boundaries, but it
  cannot yet: its reset is the *only* reset on the ALE path (`dqn_atari.make_train` calls
  `env.reset` once before its `lax.scan` and never again, `gymnax_bridge` only collapses
  flags, and `FrameStackWrapper` never calls inner reset), and its JAX-visible state is a
  one-byte dummy token, so §6.3's "`env.reset` must be safe to call every step" cannot hold
  until ALE and `AtariPreprocessing` state are externalised into the pytree.
- **`frame_stack.py`** — `step` performs a second, independent fused reset of the frame
  buffer. It must move in the same commit as the `python_env_bridge` fix, since either
  alone leaves the ALE path inconsistent.

- `gymnax_bridge.py:79` collapses to `done`, but it is the legacy compatibility layer and
  is deleted in step 10 of §9 rather than fixed.

## 9. Migration

Each step is independently valid and bisect-safe. Per repository policy the old path is
deleted in the same commit as its replacement; equivalence tests against previous
behaviour are scaffolding inside that commit, not surviving parallel paths.

1. Fix the two immediately-fixable adapter defects in §8.2 — the `brax` `terminated`
   mapping and the `mujoco_playground_bridge` `truncated` mapping — each with the test that
   pins it. Independent of everything else.
2. Migrate `ReplayBufferState.dones` to `discount` across eighteen files in three
   repositories (§8.1), as consumer-side preparation followed by one storage flip per
   repository. Independent of the port; unblocks everything after.
3. `Timestep`, `AgentProtocol`, `loop.run`, and `EnvSpec.truncation_policy`. No agent
   callers; tests use a two-state toy environment and a constant-action agent, and pin
   continuation / termination / truncation against hand-computed targets. The three
   loop-dependent adapter defects of §8.2 — `brax` `auto_reset`, `python_env_bridge`'s
   fused reset, and `frame_stack`'s — land here, because only now is there a loop to own
   the boundary.
4. Port `DQN` to `DQNAgent`; delete `dqn.make_train`; update callers and tests.
5. `double_dqn`, `dueling_dqn` — near-identical to DQN, should be close to free.
6. `sac`, `td3`, `qrc`, `greedy_ac`. `sac` and `td3` must migrate together with
   `projects/process-control-baselines` (`rl_comparison.py:156-158`,
   `declarative.py:304,317`), which calls their `make_train` directly.
7. `dqn_atari` — already half-split via `init_runner_state` / `make_train_step`.
8. `ppo` last: per-step rollout accumulation, the completion-indexed slot layout of §4.5
   with `bootstrap_value` stored forward, and the GAE change.
9. `rainbow`: the §4.5 n-step change — cumulative `D_t`, a `jax.Array` discount in
   `categorical_target_support`, deletion of the compile-time `bootstrap_discount`. It uses
   `jax_replay.per`, not `ReplayBuffer`, so it is untouched by step 2 and carries the fused
   flag until here.
10. Delete `rl_components.gym_env.GymEnv`. By now the loop is the only consumer of `env`,
    so this drops from a twelve-file change to a one-file change, and `EnvProtocol` finally
    becomes the real port.
11. Migrate the project agents that copy the skeleton (`iterated-gtd`,
    `gac-gradient-refinement`), including `comparison.py` and `discrete_comparison.py`.

PPO's per-step restructuring shifts the RNG consumption pattern, so it will not reproduce
bitwise against current baselines. Gate step 8 on metric-level equivalence, not exact
match. Steps 1 and 2 are the only ones that touch no port code; everything from step 3 on
depends on the loop existing.

## 10. Testing

The point of the port is that boundary correctness becomes unit-testable without an
environment. Per `AGENTS.md` tiers:

`small/`

- `Timestep` construction over the full cross-product of
  `(terminated, truncated, cutoff_reached, truncation_policy)`, including
  **`terminated & truncated` simultaneously**, asserting termination dominance (§4.1).
- One-step targets under continuation / termination / truncation against hand-computed
  values.
- GAE over a hand-built rollout containing one termination and one truncation, with
  `V(s_T) != V(s_0_new)`, asserting that the advantage at the truncation **depends on
  `V(s_T)` and is invariant to `V(s_0_new)`**. A test where those two values coincide, or
  where `bootstrap_value` is supplied by hand, passes even with cross-episode leakage and
  is therefore worthless.
- The n-step accumulator flushing on a truncation inside its window, asserting that each
  of the `n` flushed items receives its **own** `D_t` for horizon `m - i`, not a shared
  `gamma ** n`.
- **The §5 ordering rule, for every eligibility-trace agent** (`qrc`, `greedy_ac`): the
  update at an episode's final transition must use the trace accumulated *before* the
  boundary. Nothing in the type system prevents an agent from zeroing the trace first, so
  this is the only guard.
- Rollout index alignment for PPO: the first rollout has `NUM_STEPS - 1` valid slots and
  every subsequent rollout has `NUM_STEPS`, since `valid` is false only at global `t = 0`.

`medium/`

- `run` under `jax.jit` against a toy environment with a known optimal value function.
- The true final observation reaches the agent as `bootstrap_observation` at a boundary,
  while `observation` is the post-reset state.
- Step index equals transition count over a run containing several boundaries.
- **`jax.vmap(run, in_axes=(None, None, 0))` across seeds with randomised episode
  lengths**, asserting pytree shape stability when seeds hit boundaries at different
  iterations, and that per-seed schedules stay aligned to per-seed experience.

Verification for the step-2 buffer commit specifically: `test_rl_components_buffer.py`,
`test_rl_components_buffer_jit.py`, the Atari benchmark fixture, and the `iterated-gtd` and
`gac-gradient-refinement` suites must all pass in that commit, not merely the eight library
agents. The commit is behaviour-preserving, so every existing assertion should hold
unchanged except `test_dones_dtype_is_bool`, which is replaced by a `float32` discount
assertion.

Also missing: a check at the final scan iteration `N-1`, which closes transition `N-2` and
takes an action whose transition never closes, confirming no out-of-bounds write or stale
slot.

`large/`

- Per-agent metric-level equivalence against the pre-migration `make_train` for each step
  of §9 — **on environments that do not truncate.** Pre-migration code is defective at
  truncation, so equivalence there would be evidence of a *failed* port. Truncation
  correctness must instead be verified against an analytically tractable MDP (a random walk
  with a known closed-form value under cutoff), not against the old behaviour.

## 11. Open Questions

Resolved by review:

- **O1 (reset ordering) — resolved.** Loop-owned autoreset, §6.2. Step-index dilation under
  `vmap` decided it against the non-fused alternative.
- **O2 (`truncation_policy` placement) — resolved.** Default on `EnvSpec`, per-run override
  on `run`, §4.4.
- **O4 (`lax.cond` vs ADR 004) — resolved, moot.** `cond` vectorises to `select` under
  `vmap`, so it buys nothing here; use masking per ADR 004, §6.3.
- **O7 (off-policy correction) — resolved.** See below.

Open:

- **O3.** Should `Timestep` carry `terminated` / `truncated` explicitly despite being
  recoverable? Costs port minimality, buys agent-side clarity and cleaner metrics.
- **O5.** Does any planned algorithm need a target discount differing from the evaluation
  discount, now that `ADDITIONAL_DISCOUNT` is deleted rather than reinterpreted (§6)?
- **O6.** Per-step `metrics: dict[str, jax.Array]` stacks every key over the full horizon.
  At 200k steps under `vmap` across 32-64 seeds this allocates gigabytes of device memory
  and may be the binding constraint on sweep size. A windowing or subsample reduction
  inside `run` would mirror the thesis `Collector` samplers at the cost of loop
  configuration. This is now a capacity question, not a taste question.
- **O7 — resolved.** Off-policy correction (importance sampling, Retrace, V-trace) keeps
  behaviour-policy probabilities in `AgentState`, not on `Timestep`. `Timestep` models the
  environment-to-agent boundary, and the loop has no access to action distributions;
  `mu(a_t | s_t)` is recorded alongside `(s_t, a_t)` by the same mechanism and retrieved
  when the transition closes.
