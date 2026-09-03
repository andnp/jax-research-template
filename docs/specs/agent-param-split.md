# Config/Hypers Split Classification — `libs/rl-agents`

> Terminology: **hypers** here means swept *hyperparameters* — learning rate,
> discount, tau, epsilon-schedule scalars. It never means learned network
> weights, which live in agent state as `train_state.params` / `target_params`
> and are already traced. Do not name the dynamic pytree `params`.

> Status: the split described here is IMPLEMENTED for every agent. Each has a
> `*Hypers` chex_struct and a `*_hypers(config)` builder; the `AgentProtocol`
> agents carry it as a `hypers` field on their state, while `ppo` and `rainbow`
> take it as a traced argument to `train` rather than being ported to the port.
> `AgentProtocol` itself was left unchanged. Two fields remain unsweepable by
> necessity: `greedy_ac.ACTOR_PERCENTILE` and `greedy_ac.UNIFORM_WEIGHT`.

> Snapshot notice: this document reflects `libs/rl-agents` (and the referenced
> parts of `libs/rl-components`) as read at commit `2ec67ab5`. Another session
> was actively editing `libs/rl-agents` and `libs/rl-components` while this
> analysis was taken — treat any surprising discrepancy against the live tree
> as the tree having moved, not as an error in this document. This is a
> read-only analysis; no source files were modified to produce it.

Directory confirmed (`libs/rl-agents/src/rl_agents/`): `dqn.py`, `double_dqn.py`, `double_q.py`, `dueling_dqn.py`, `dqn_atari.py`, `rainbow.py`, `qrc.py`, `ppo.py`, `sac.py`, `sac_rc.py`, `td3.py`, `greedy_ac.py`, plus the shared `q_networks.py` (no config, network builders only). No file looked mid-edit at read time (no conflict markers, no truncated syntax), but another session is active in this tree — treat this as a snapshot.

**Structural split first**: the DQN family (`dqn`, `double_dqn`/`double_q`, `dueling_dqn`, `dqn_atari`) already implements the new `AgentProtocol` (`init`/`step`, config bound in `__init__`, state is a `chex_struct` pytree). `rainbow`, `qrc`, `ppo`, `sac`, `sac_rc`, `td3`, `greedy_ac` are all still the older `make_train(config, env, env_params) -> train(rng)` closure style — a single monolithic `lax.scan`, not the two-method port. This matters for the "does `make_train` close over config" question below: for the still-migrating half, the entire config is closed over the outer `train` closure at trace time, by construction — a hypers split there requires either threading hypers through the scan carry or rewriting them onto `AgentProtocol` first.

---

## dqn.py — `DQNConfig` (`libs/rl-agents/src/rl_agents/dqn.py:42`)

| Field | Class | Justification |
|---|---|---|
| LR | DYNAMIC | passed straight into `optax.adam(config.LR)`; a scalar multiplier |
| BUFFER_SIZE | STATIC | sizes `ReplayBuffer(capacity=...)`, an array shape at construction |
| BATCH_SIZE | STATIC | passed to `buffer.sample(..., config.BATCH_SIZE)`, which fixes the sampled batch's leading array dimension |
| TOTAL_TIMESTEPS | UNCERTAIN | only used as a divisor inside the epsilon-decay arithmetic (`step_index / (TOTAL_TIMESTEPS * EPSILON_FRACTION)`) in this file — looks DYNAMIC here — but it is also the `lax.scan` length in the (not-yet-existing for this agent) training loop; since the agent itself never uses it as a bound, classify DYNAMIC for this module but flag: the training-loop caller (`rl_components.loop.run`) likely needs it as a static scan length, which is a cross-boundary conflict, not an in-agent one |
| LEARNING_STARTS | DYNAMIC | only used in a comparison `step_index > config.LEARNING_STARTS` feeding `lax.cond`'s predicate; the predicate is traced, not branched in Python |
| TRAIN_FREQUENCY | DYNAMIC | only in `step_index % config.TRAIN_FREQUENCY == 0`, a traced modulo comparison |
| TARGET_NETWORK_FREQUENCY | DYNAMIC | same pattern, `step_index % config.TARGET_NETWORK_FREQUENCY == 0` |
| TAU | DYNAMIC | scalar in the polyak-average arithmetic |
| EPSILON_START | DYNAMIC | scalar in epsilon schedule arithmetic |
| EPSILON_END | DYNAMIC | scalar in epsilon schedule arithmetic (also `jnp.maximum` floor) |
| EPSILON_FRACTION | DYNAMIC | scalar divisor in schedule arithmetic |
| ENV_NAME | STATIC | selects the environment at construction (Python-level, outside JAX entirely) |
| SEED | STATIC | Python-level PRNG seed used to build the initial key before `init` is called — never a traced value inside `step` |
| NETWORK_PRESET | STATIC | `Literal["mlp","nature_cnn"]` read by `make_q_network` via Python `if` branches at trace time — picks which network class gets built, i.e. the compiled kernel itself |

Entry point: `DQNAgent.__init__` binds `self.config = config` once; `step`/`init` both read `self.config` closed over the Python object (per the module docstring: "the agent object itself is static under jit"). A Config/Hypers split here means `DQNAgent.__init__` takes only the STATIC fields (Config), and every DYNAMIC field must move onto `DQNAgentState` (or a sibling hypers pytree threaded into `step`) since the agent object cannot carry a traced leaf.

## double_dqn.py / double_q.py — `DoubleDQNConfig` (`double_dqn.py:20`), shared update in `DoubleQAgent`/`DoubleQConfig` Protocol (`double_q.py:74`, `double_q.py:196`)

Identical field set and classifications to `DQNConfig` above (same names, same usage sites — `double_q.py` is the shared engine both `double_dqn.py` and `dueling_dqn.py` bind into). No inconsistency versus `dqn.py`.

`DoubleQConfig` is a `typing.Protocol` (structural), not a `chex_struct` — it's the interface the shared update code reads, not a pytree itself; the two concrete `chex_struct` configs (`DoubleDQNConfig`, `DuelingDQNConfig`) each separately satisfy it. Any Config/Hypers split needs to happen at the concrete `chex_struct` classes; the Protocol would need a corresponding split into `StaticConfig`/`DynamicHypers` protocols, or be dropped in favor of two protocols.

`DoubleQAgent.__init__(self, config, network_factory)` binds both the config and a `network_factory` closure; `network_factory` itself closes over the concrete config (see `double_dqn.py:49`, `dueling_dqn.py:95`) to decide `NETWORK_PRESET` at construction — reinforcing that `NETWORK_PRESET` is STATIC (it decides which network-factory closure gets built, prior to any tracing).

## dueling_dqn.py — `DuelingDQNConfig` (`dueling_dqn.py:30`)

Same field set/classifications as `DQNConfig`. One nuance: `NETWORK_PRESET` here only supports `"mlp"` — `"nature_cnn"` raises `ValueError` at construction (`_make_dueling_q_network`, `dueling_dqn.py:76-79`). Still STATIC, same reasoning; the raise happens at Python level via `_make_dueling_q_network`, called from `__init__` (not `step`), so it's resolved once per static-config batch, exactly where a vmap zone boundary should be.

## dqn_atari.py — `DQNAtariConfig` (`dqn_atari.py:54`) + `DQNAtariRuntimeConfig` (`dqn_atari.py:71`)

This module has real STATIC/DYNAMIC subtlety — several fields that look like "just a schedule parameter" are baked into Python integers at `__init__` time and never retraced.

`DQNAtariConfig`:

| Field | Class | Justification |
|---|---|---|
| REPLAY_CAPACITY | STATIC | `ReplayBuffer(capacity=self.config.REPLAY_CAPACITY, ...)` — array shape |
| MIN_REPLAY_CAPACITY_FRACTION | STATIC | consumed only by `dqn_zoo_atari_min_replay_capacity`, called once in `__init__` (`self.min_replay_capacity = ...`) to produce a **Python int** stored on the agent object — never touches a traced value; changing it changes a value baked into the compiled `step` via closure |
| BATCH_SIZE | STATIC | sizes the sampled minibatch's leading dimension via `buffer.sample(..., config.BATCH_SIZE)` |
| NUM_ACTION_REPEATS | STATIC | pure Python-int arithmetic divisor in `dqn_zoo_atari_frames_to_env_steps`, called only at `__init__`/module level to convert frame-counted periods to env-step periods — this integer conversion must not be restaged per iteration (explicitly called out in the module docstring) |
| TARGET_NETWORK_UPDATE_PERIOD_FRAMES | STATIC | resolved once via `dqn_zoo_atari_target_update_period_env_steps` into `self.target_update_period_env_steps` (a Python int), then used in `step_index % self.target_update_period_env_steps` — the modulo constant itself is fixed at trace time, not a traced leaf |
| LEARN_PERIOD_FRAMES | STATIC | same pattern → `self.learn_period_env_steps`, a construction-time Python int |
| LEARNING_RATE | DYNAMIC | flows into `optax.rmsprop(learning_rate=...)` as a scalar |
| OPTIMIZER_EPSILON | DYNAMIC | scalar into `optax.rmsprop(eps=...)` |
| RMSPROP_DECAY | DYNAMIC | scalar into `optax.rmsprop(decay=...)` |
| RMSPROP_CENTERED | STATIC | `bool` passed to `optax.rmsprop(centered=...)`, which is a Python-level branch inside optax selecting a different update rule/state shape, not a traced value |
| EXPLORATION_EPSILON_BEGIN | DYNAMIC | scalar in `_exploration_epsilon` arithmetic (`config.EXPLORATION_EPSILON_BEGIN + ...`) |
| EXPLORATION_EPSILON_END | DYNAMIC | scalar in same arithmetic |
| EXPLORATION_EPSILON_DECAY_FRAME_FRACTION | STATIC | consumed only by `dqn_zoo_atari_exploration_decay_env_steps`, called once at `__init__` to produce `self.exploration_decay_env_steps` (Python int) — same "resolved once, baked into closure" pattern |

`DQNAtariRuntimeConfig`:

| Field | Class | Justification |
|---|---|---|
| TOTAL_TRAIN_ENV_STEPS | STATIC | this is the `lax.scan` length for the training loop that will eventually call this agent; also used at `__init__` (via `runtime_config`) to compute `self.exploration_decay_env_steps` — a shape/loop-bound quantity |
| SEED | STATIC | Python-level PRNG seed, same as `dqn.py` |
| EVAL_EXPLORATION_EPSILON | DYNAMIC | only ever a scalar used at evaluation time (not read inside `step` at all currently — dead in this module, but by shape a scalar) |

This module is the clearest illustration of "looks dynamic but is static in practice": `MIN_REPLAY_CAPACITY_FRACTION`, `NUM_ACTION_REPEATS`, `TARGET_NETWORK_UPDATE_PERIOD_FRAMES`, `LEARN_PERIOD_FRAMES`, and `EXPLORATION_EPSILON_DECAY_FRAME_FRACTION` are all fractional/period-shaped fields that a naive glance would vmap, but `__init__` (`dqn_atari.py:238-243`) immediately resolves each through Python-integer division into a plain Python int stored as `self.<name>`, and `step`/`_exploration_epsilon` only ever read `self.<name>`, never `self.config.<name>` again. Vmapping the raw config field would do nothing (the resolved int is already baked into the closure) and worse, would silently decouple the runtime int from any override.

`DQNAtariAgent.__init__(self, config, runtime_config)` (line 229) is exactly the pattern the Config/Hypers split targets: two config objects go in, five derived Python ints come out and get stored on `self`. A refactor must decide whether the *derived* ints (`self.min_replay_capacity` etc.) belong to the new static `Config`, computed once, or whether they need to be recomputed per batch element if any of their static inputs varies across points in the same vmap zone (they can't vary — by definition all points in a batch share the same static config).

## rainbow.py — `RainbowConfig` (`rainbow.py:25`) + `RainbowRuntimeConfig` (`rainbow.py:44`)

Still the `make_train` closure style, not `AgentProtocol`. `make_train(config, runtime_config, env, env_params)` (line 537) closes over both configs directly into the returned `train(rng)`; `make_train_step` (line 389-397) additionally closes `support = rainbow_support(config)` (an array built from `V_MIN`/`V_MAX`/`NUM_ATOMS`) and `bootstrap_discount = config.ADDITIONAL_DISCOUNT**config.N_STEP` at trace-setup time, both Python-level.

| Field | Class | Justification |
|---|---|---|
| REPLAY_CAPACITY | STATIC | `init_per_buffer(replay_prototype, config.REPLAY_CAPACITY)` — buffer array shape |
| MIN_REPLAY_CAPACITY_FRACTION | STATIC | resolved via `rainbow_zoo_atari_min_replay_capacity`, called inside `_rainbow_zoo_atari_should_learn_array` every step but as pure Python-int arithmetic on `config` (not a traced field) before being cast with `jnp.asarray` — same "Python-int-then-cast" shape as `dqn_atari` |
| BATCH_SIZE | STATIC | `per_sample(..., batch_size=config.BATCH_SIZE, ...)` — fixes sampled batch shape |
| NUM_ACTION_REPEATS | STATIC | frame→env-step conversion divisor, Python arithmetic only |
| TARGET_NETWORK_UPDATE_PERIOD_FRAMES | STATIC | resolved to a Python int via `rainbow_zoo_atari_target_update_period_env_steps`, feeds `_rainbow_zoo_atari_should_update_target_array`'s modulo constant |
| LEARN_PERIOD_FRAMES | STATIC | same, feeds the learn-period modulo constant |
| LEARNING_RATE | DYNAMIC | scalar into `optax.rmsprop(learning_rate=...)` |
| OPTIMIZER_EPSILON | DYNAMIC | scalar into `optax.rmsprop(eps=...)` |
| RMSPROP_DECAY | DYNAMIC | scalar into `optax.rmsprop(decay=...)` |
| RMSPROP_CENTERED | STATIC | bool selecting optax's centered-vs-plain RMSProp code path |
| ADDITIONAL_DISCOUNT | UNCERTAIN — used both as a plain scalar discount factor (arithmetic, DYNAMIC-shaped) in `categorical_target_probabilities`/`_materialize_n_step_transitions`, **and** exponentiated by the STATIC `N_STEP` at trace-setup time into `bootstrap_discount = config.ADDITIONAL_DISCOUNT**config.N_STEP` (line 399, computed once outside the scan, closed over as a Python/array constant). If `ADDITIONAL_DISCOUNT` is vmapped, that closure-time exponentiation must move inside the scan body instead of being precomputed once — a real split hazard, not merely cosmetic |
| N_STEP | STATIC | sizes every n-step accumulator array: `_init_n_step_accumulator` does `jnp.zeros((config.N_STEP, *prototype.obs.shape))`; also the loop bound in `indices = jnp.arange(config.N_STEP)` and the trip count implicit in `_shift_n_step_accumulator_left`'s slicing — an array-shape and reshape-bound field, textbook STATIC |
| NUM_ATOMS | STATIC | sets `RainbowNatureNetwork(num_atoms=...)` — a Flax module field controlling output layer width/reshape (`value_logits.reshape(..., (1, self.num_atoms))`) — network architecture |
| V_MIN | DYNAMIC | scalar bound passed to `jnp.linspace(config.V_MIN, config.V_MAX, config.NUM_ATOMS)` to build the support array — the **array's length** is `NUM_ATOMS` (STATIC), but `V_MIN`/`V_MAX` only set its numeric endpoints, so they can vary without changing shape |
| V_MAX | DYNAMIC | same as V_MIN |

`RainbowRuntimeConfig.TOTAL_TRAIN_ENV_STEPS` — STATIC (scan length, `jnp.arange(total_env_steps)` at `make_train` line 548); `SEED` — STATIC (Python PRNG seed).

Cross-agent note: `RMSPROP_CENTERED` is STATIC here and in `dqn_atari`, consistent. `ADDITIONAL_DISCOUNT` has no counterpart named field elsewhere (other agents use `GAMMA`, which is uniformly DYNAMIC — see below) — it is the one config field in the whole survey that is UNCERTAIN by the stated rule ("used both as a divisor and as a range() bound" analog: used as arithmetic scalar and exponentiated by a static int at trace-setup).

`make_train`/`make_train_step` close over `config`/`runtime_config` directly (no agent object, no `AgentStateT`); state lives in a bare `NamedTuple` `RunnerState`. This is the "monolithic scan closure" pattern — a Config/Hypers split requires either (a) migrating rainbow onto `AgentProtocol` first, or (b) accepting hypers as an extra scan input alongside `RunnerState`, which the current `train_step(runner_state, step_index)` signature (no params argument) does not support.

## qrc.py — `QRCConfig` (`qrc.py:34`)

`make_train` closure style.

| Field | Class | Justification |
|---|---|---|
| LR | DYNAMIC | `optax.adam(config.LR)` scalar |
| BUFFER_SIZE | STATIC | `ReplayBuffer(config.BUFFER_SIZE, ...)` capacity/shape |
| BATCH_SIZE | STATIC | `buffer.sample(buffer_state, _rng, config.BATCH_SIZE)` — batch shape |
| TOTAL_TIMESTEPS | STATIC | `lax.scan` trip count: `jnp.arange(config.TOTAL_TIMESTEPS)` at line 242 — also divides the epsilon schedule, but the scan-length usage alone makes it STATIC (a genuine loop bound, unlike `dqn.py`'s `DQNConfig.TOTAL_TIMESTEPS` which in that file only appears in schedule arithmetic — see cross-agent note below) |
| LEARNING_STARTS | DYNAMIC | comparison only, `t > config.LEARNING_STARTS`, feeds `lax.cond` |
| TRAIN_FREQUENCY | DYNAMIC | `t % config.TRAIN_FREQUENCY == 0`, traced modulo |
| GAMMA | DYNAMIC | scalar, `discount = config.GAMMA * (1.0 - done)` |
| EPSILON_START | DYNAMIC | schedule scalar |
| EPSILON_END | DYNAMIC | schedule scalar |
| EPSILON_FRACTION | DYNAMIC | schedule scalar divisor |
| BETA | DYNAMIC | scalar weight on h-head L2 regularizer, `qrc_loss_batch(..., beta=config.BETA)` |
| ENV_NAME | STATIC | environment selection, Python-level |
| SEED | STATIC | PRNG seed, Python-level |

Cross-agent inconsistency: `TOTAL_TIMESTEPS` is STATIC in `qrc.py` (it is literally the scan length passed to `jnp.arange`/`lax.scan`) but DYNAMIC in `dqn.py`'s `DQNAgent` (where it only appears inside epsilon-schedule division, because the scan itself lives outside the agent, in `rl_components.loop.run`, per the `dqn.py` docstring's explicit statement that the agent "owns no ... `lax.scan`"). This is the single most important cross-agent field: **the same name means different things depending on whether the loop-length responsibility sits inside the agent module (old style) or in the shared loop (`AgentProtocol` style)**. Once `qrc` is migrated to `AgentProtocol`, `TOTAL_TIMESTEPS` should become DYNAMIC there too (used only in the epsilon-fraction division), matching `dqn.py`.

## ppo.py — `PPOConfig` (`libs/rl-components/src/rl_components/types.py:5`, imported by `ppo.py`)

Note: `PPOConfig` lives in `rl_components.types`, not `rl_agents` — it's the one config in this survey defined outside `rl-agents`. `make_train` closure style.

| Field | Class | Justification |
|---|---|---|
| LR | DYNAMIC | scalar into `optax.adam(config.LR, eps=1e-5)` |
| NUM_STEPS | STATIC | `lax.scan(_env_step, runner_state, None, config.NUM_STEPS)` — trip count/rollout-buffer length; also the reshape divisor `[config.NUM_MINIBATCHES, -1]` size implicitly derives from it |
| TOTAL_TIMESTEPS | STATIC | `num_updates = config.TOTAL_TIMESTEPS // config.NUM_STEPS`, then `lax.scan(_update_step, ..., num_updates)` — a scan trip count computed via Python integer division |
| UPDATE_EPOCHS | STATIC | `lax.scan(_update_epoch, update_state, None, config.UPDATE_EPOCHS)` — scan trip count |
| NUM_MINIBATCHES | STATIC | reshape target: `.reshape([config.NUM_MINIBATCHES, -1] + ...)` — directly sets an array shape |
| GAMMA | DYNAMIC | scalar in GAE recursion (`delta = reward + config.GAMMA * next_value * not_done - value`) |
| GAE_LAMBDA | DYNAMIC | scalar in GAE recursion |
| CLIP_EPS | DYNAMIC | scalar clip bound, `jnp.clip(ratio, 1.0 - config.CLIP_EPS, 1.0 + config.CLIP_EPS)` |
| ENT_COEF | DYNAMIC | scalar loss-coefficient |
| VF_COEF | DYNAMIC | scalar loss-coefficient |
| MAX_GRAD_NORM | DYNAMIC | scalar into `optax.clip_by_global_norm(config.MAX_GRAD_NORM)` — this is a numeric threshold, not a shape, despite living inside an optax transformation chain |
| REWARD_SCALE | DYNAMIC | scalar multiplier on reward; validated once at `make_train` entry (`if not math.isfinite(...) or <= 0`) which is a Python-level guard outside any trace, not a trace-time structural decision — so still DYNAMIC, just validated eagerly |
| NORMALIZE_OBSERVATIONS | STATIC | `bool` read via Python `if not enabled: return obs` inside `_maybe_normalize_observation`/`_maybe_update_observation_norm_state` (lines 87-90, 100-103) — a genuine Python-level branch selecting whether normalization code runs at all, taken at trace time |
| OBS_NORM_EPS | DYNAMIC | scalar inside `jnp.sqrt(variance + eps)` |
| OBS_NORM_CLIP | DYNAMIC | scalar clip bound |
| ENV_NAME | STATIC | environment selection |
| SEED | STATIC | PRNG seed |

`ppo.py` is the only agent whose entry point closes over a boolean-flag structural branch (`NORMALIZE_OBSERVATIONS`) outside the network/shape category — worth flagging since it's easy to miss under the "shape/branch/loop-bound" framing: it's a Python `if`, so STATIC by the stated rule even though it reads like a plain on/off toggle.

## sac.py — `SACConfig` (`sac.py:16`)

`make_train` closure style.

| Field | Class | Justification |
|---|---|---|
| LR | DYNAMIC | scalar, shared by `optax.adam(config.LR)` for actor, critic, and alpha optimizers |
| BUFFER_SIZE | STATIC | `ReplayBuffer(config.BUFFER_SIZE, ...)` capacity |
| BATCH_SIZE | STATIC | `buffer.sample(..., config.BATCH_SIZE)`; also `jax.random.split(_rng, config.BATCH_SIZE)` sizes the per-sample PRNG key array |
| TOTAL_TIMESTEPS | STATIC | `lax.scan(_update_step, ..., jnp.arange(config.TOTAL_TIMESTEPS))` — trip count |
| LEARNING_STARTS | DYNAMIC | comparison only, `t < config.LEARNING_STARTS` |
| TRAIN_FREQUENCY | DYNAMIC | `t % config.TRAIN_FREQUENCY == 0` |
| GAMMA | DYNAMIC | scalar, `discount = config.GAMMA * (1.0 - done)` |
| TAU | DYNAMIC | scalar polyak coefficient |
| ALPHA | UNCERTAIN | declared in the config but **not referenced anywhere in `make_train`/`_update_step`/`_do_train`** — the entropy temperature comes from `alpha_state.params["log_alpha"]`, initialized to `jnp.zeros(1)` regardless of `config.ALPHA`. This is a dead field, not clearly DYNAMIC or STATIC because it is not consumed in the traced path at all — flag for whoever owns this file about whether it's dead code or a missed wiring |
| TARGET_ENTROPY | STATIC | read once via Python `if config.TARGET_ENTROPY is None: target_entropy = -float(action_dim) else: ...` (lines 116-119) — a Python-level branch at `make_train`-construction time, resolved before any tracing, producing a Python float closed over by `_alpha_loss_fn` |
| ENV_NAME | STATIC | environment selection |
| SEED | STATIC | PRNG seed |

## sac_rc.py — `SACRCConfig` (`sac_rc.py:44`)

Same field set/classifications as `SACConfig` for the shared fields (`LR`, `BUFFER_SIZE`, `BATCH_SIZE`, `TOTAL_TIMESTEPS`, `LEARNING_STARTS`, `TRAIN_FREQUENCY`, `GAMMA`, `TARGET_ENTROPY`, `ENV_NAME`, `SEED`), plus:

| Field | Class | Justification |
|---|---|---|
| ALPHA | UNCERTAIN | same as `sac.py` — the dead-field concern applies here too (`alpha` at runtime comes from `alpha_state.params["log_alpha"]`, initialized to `jnp.zeros(1)`, not from `config.ALPHA`) |
| BETA | DYNAMIC | scalar weight on h-head L2 regularizer, `sac_rc_loss_batch(..., beta=config.BETA)` — mirrors `qrc.py`'s `BETA` |

Note `sac_rc.py` has **no `TAU`** field at all — unlike `sac.py`, it never soft-updates a critic target (the module docstring: "There is no target network... every bootstrap uses the online critic parameters"). This is the clearest same-named-field-missing-elsewhere case: `TAU` exists in `sac.py`/`td3.py`/`dqn.py` family but is architecturally absent from `sac_rc.py` and `qrc.py`, because both are target-network-free gradient-TD variants.

## td3.py — `TD3Config` (`td3.py:17`)

`make_train` closure style, but notably `make_train` itself (not just `train`) precomputes shape-derived quantities: `action_dim`, `obs_dim`, `action_shape`, and builds `actor`, `critic`, `buffer` at `make_train`-construction time (lines 140-149), before `train(rng)` is even called — closer in spirit to the `AgentProtocol` split than the other closure-style agents.

| Field | Class | Justification |
|---|---|---|
| LR | DYNAMIC | scalar, `optax.adam(config.LR)` for both actor and critic |
| BUFFER_SIZE | STATIC | `ReplayBuffer(config.BUFFER_SIZE, ...)` capacity |
| BATCH_SIZE | STATIC | `buffer.sample(..., config.BATCH_SIZE)` batch shape |
| TOTAL_TIMESTEPS | STATIC | `lax.scan(_update_step, ..., jnp.arange(config.TOTAL_TIMESTEPS))` — trip count |
| LEARNING_STARTS | DYNAMIC | comparisons only, `t < config.LEARNING_STARTS`, `t > config.LEARNING_STARTS` |
| TRAIN_FREQUENCY | DYNAMIC | `t % config.TRAIN_FREQUENCY == 0` |
| GAMMA | DYNAMIC | scalar, `discount = config.GAMMA * (1.0 - done)` |
| TAU | DYNAMIC | scalar polyak coefficient, `_soft_update(config.TAU, ...)` |
| POLICY_DELAY | DYNAMIC | used only in `t % config.POLICY_DELAY == 0`, a traced modulo feeding `lax.cond` — classic "looks like it should be static (it's an integer period) but is only ever a traced comparand," same shape as `TRAIN_FREQUENCY`/`TARGET_NETWORK_FREQUENCY` elsewhere |
| EXPLORATION_NOISE | DYNAMIC | scalar std multiplier on action noise |
| POLICY_NOISE | DYNAMIC | scalar std multiplier on target-policy smoothing noise |
| NOISE_CLIP | DYNAMIC | scalar clip bound, `jnp.clip(noise, -config.NOISE_CLIP, config.NOISE_CLIP)` |
| ENV_NAME | STATIC | environment selection |
| SEED | STATIC | PRNG seed |

Cross-agent note: `POLICY_DELAY` is the TD3-specific analog of `TARGET_NETWORK_FREQUENCY` (DQN family) and both classify DYNAMIC by the same reasoning (traced modulo comparand, not a Python-level branch or shape) — worth double-checking against intuition since "delay"/"frequency" period integers read as if they should gate compiled structure, but here they only gate a runtime `lax.cond`, so the compiled graph is identical regardless of value.

## greedy_ac.py — `GACConfig` (`greedy_ac.py:42`)

`make_train` closure style; largest and most shape-sensitive config in the survey.

| Field | Class | Justification |
|---|---|---|
| LR | DYNAMIC | scalar into `optax.adamw(learning_rate=config.LR, ...)` for the critic |
| ACTOR_LR | DYNAMIC | scalar into `optax.adamw(learning_rate=config.ACTOR_LR, ...)` for the actor |
| BUFFER_SIZE | STATIC | `ReplayBuffer(config.BUFFER_SIZE, ...)` capacity |
| BATCH_SIZE | STATIC | `buffer.sample(..., config.BATCH_SIZE)`; also sizes `jax.random.split(next_rng, config.BATCH_SIZE)` and `jax.random.split(proposal_rngs, config.BATCH_SIZE)` — key-array shapes |
| TOTAL_TIMESTEPS | STATIC | `lax.scan(_update_step, ..., jnp.arange(config.TOTAL_TIMESTEPS))` — trip count |
| LEARNING_STARTS | DYNAMIC | comparison only, `t > config.LEARNING_STARTS`/`t < config.LEARNING_STARTS` |
| TRAIN_FREQUENCY | DYNAMIC | `t % config.TRAIN_FREQUENCY == 0` |
| GAMMA | DYNAMIC | scalar, `discount = config.GAMMA * (1.0 - done)` |
| TAU | UNCERTAIN | declared in the config (and docstring: "Target network polyak averaging coefficient") but **not referenced anywhere in `make_train`/`_do_train`** — this agent has no target network in its code path at all (matches `sac_rc`/`qrc`'s "no target network" design, but here the field is simply dead, unlike `sac_rc.py` which omits the field outright). Same dead-field category as `sac.py`'s `ALPHA` |
| NUM_SAMPLES | STATIC | directly sizes arrays: `total = B * num_samples`; `flat_proposals = proposals.reshape(total, action_dim)`; `q_values = flat_q.reshape(B, num_samples)` — a reshape/shape-determining integer, unambiguous STATIC |
| ACTOR_PERCENTILE | UNCERTAIN | used as `k = max(1, min(int(actor_percentile * num_samples), num_samples))` (`greedy_ac.py:398`), i.e. it is multiplied by the STATIC `NUM_SAMPLES` and **cast through Python `int()`** to become the `k` argument of `jax.lax.top_k(q_values, k)` — `top_k`'s `k` is a static (shape-determining) argument to the primitive. So despite being a "fraction" that reads as a tunable dial, it currently must be a Python float at trace time, making it STATIC by necessity (it sets `top_k`'s output shape), not DYNAMIC. This is exactly the "genuinely ambiguous" case: semantically a hyperparameter one wants to sweep continuously, but mechanically fixed to Python-int arithmetic feeding a shape argument |
| UNIFORM_WEIGHT | UNCERTAIN | same pattern: `n_unif = max(1, int(num_samples * uniform_weight))` (`greedy_ac.py:375`) is Python-int arithmetic that sizes `jax.random.uniform(unif_key, (n_unif, action_dim), ...)`'s array shape and the split point in `jnp.concatenate([unif, prop], axis=0)`. Also STATIC-by-necessity for the same shape reason as `ACTOR_PERCENTILE` |
| ENTROPY_WEIGHT | DYNAMIC | scalar loss-coefficient, `loss = nll - entropy_weight * ent` |
| NUM_RAND_ACTIONS | STATIC | sizes arrays in `_select_best_next_action`: `total = B * N`, `flat_proposals = proposals.reshape(total, action_dim)`, `q_values = flat_q.reshape(B, N)` — same reshape-bound pattern as `NUM_SAMPLES` |
| TRAINING_SIGMA_MIN | DYNAMIC | scalar floor added to `std`, `std = jnp.exp(log_std) + sigma_min` |
| INFERENCE_SIGMA_MIN | DYNAMIC | same, scalar floor, used at inference-time action selection |
| H_REGULARIZATION | DYNAMIC | scalar weight on h-head L2 term, `_batch_qrc_loss(..., reg_weight)` |
| HIDDEN_SIZE | STATIC | `nn.Dense(self.hidden_size)` field on both `GACActor`/`GACCritic` — network width, architecture |
| ENV_NAME | STATIC | environment selection |
| SEED | STATIC | PRNG seed |

`ACTOR_PERCENTILE` and `UNIFORM_WEIGHT` are this survey's second and third UNCERTAIN entries, and unlike `RainbowConfig.ADDITIONAL_DISCOUNT` (arithmetic-vs-exponent ambiguity), theirs is a **shape-vs-hyperparameter** ambiguity: the current code forces them static via `int()`, but nothing about their semantics requires that — a refactor could re-derive `k`/`n_unif` from a fixed `NUM_SAMPLES` and a separately-vmapped fractional weight by rounding differently (e.g. computing a soft top-k mask), which would be a real design decision for whoever does the Config/Hypers split, not just a classification call.

---

## Cross-agent inconsistency summary

- **`TOTAL_TIMESTEPS`**: STATIC everywhere it is the module's own `lax.scan` trip count (`qrc`, `ppo` — via `TOTAL_TIMESTEPS // NUM_STEPS`, `sac`, `sac_rc`, `td3`, `greedy_ac`), but DYNAMIC in `dqn.py`'s `DQNConfig` because that agent has been migrated to `AgentProtocol` and no longer owns its own scan — the loop length lives in the shared `rl_components.loop.run`, and `TOTAL_TIMESTEPS` only survives inside `DQNConfig` for epsilon-schedule arithmetic. This is the single field whose classification will flip for every remaining closure-style agent the moment it's migrated to `AgentProtocol` — worth calling out explicitly to whoever plans the migration order.
- **`TRAIN_FREQUENCY` / `TARGET_NETWORK_FREQUENCY` / `POLICY_DELAY` / `LEARN_PERIOD_FRAMES`+`TARGET_NETWORK_UPDATE_PERIOD_FRAMES`**: all "period" integers, but split across two different mechanisms. In the DQN family and `sac`/`sac_rc`/`td3`/`greedy_ac`, these are DYNAMIC (bare `%`/comparison feeding `lax.cond`, no Python branch, no shape). In `dqn_atari`/`rainbow`, the *frame-counted* variants (`LEARN_PERIOD_FRAMES`, `TARGET_NETWORK_UPDATE_PERIOD_FRAMES`) are STATIC because they're resolved through Python-integer division into stored ints at construction time (`self.learn_period_env_steps` etc.), and only those derived ints are used in the traced modulo. So the same *conceptual* field (a training period) is DYNAMIC in one agent and STATIC in another purely because of an intervening unit conversion — not because the underlying semantics differ.
- **`ALPHA`**: dead/unwired in both `sac.py` and `sac_rc.py` — the runtime entropy coefficient always comes from the learned `log_alpha` parameter initialized to `jnp.zeros(1)`, never from `config.ALPHA`. Flagged UNCERTAIN in both.
- **`TAU`**: present and load-bearing in `dqn`/`double_dqn`/`double_q`/`dueling_dqn` (soft update every `TARGET_NETWORK_FREQUENCY` steps), `sac`, `td3`; declared but dead in `greedy_ac`; absent entirely from `sac_rc`/`qrc` (no target network at all, by design). Not itself ambiguous per-agent, but the presence/absence pattern is worth surfacing since a shared `hypers` pytree schema across agents cannot uniformly include `TAU`.
  - Correction/nuance: `dqn_atari.py`'s target update (`dqn_atari.py:340-344`) is a hard copy (`lambda: train_state.params`) gated on `step_index % self.target_update_period_env_steps == 0`, not a polyak/`TAU`-weighted average — `DQNAtariConfig` has no `TAU` field at all, matching that. `rainbow.py` likewise hard-copies (`rainbow.py:498-502`) with no `TAU` field. So `TAU` is present in `dqn`/`double_dqn`/`double_q`/`dueling_dqn` (soft update) but absent from `dqn_atari` and `rainbow` (both hard-copy).
- **`RMSPROP_CENTERED`**: STATIC consistently in both places it appears (`dqn_atari`, `rainbow`) — no inconsistency, but worth noting since it's a `bool` optax structural flag, an easy field to misclassify as "just a knob."
- **`NETWORK_PRESET`**: STATIC consistently across `dqn`, `double_dqn`, `dueling_dqn` — all read via `make_q_network`/`_make_dueling_q_network`'s Python `if` at construction. `dqn_atari` and `rainbow` don't have this field at all (they hardcode `NatureQNetwork`/`RainbowNatureNetwork`), so there's no inconsistency, just architectural asymmetry.

## `make_train`/entry-point closure-over-config summary

- **Already `AgentProtocol`** (config closed over `self.config` on a static agent object, not over a `train` closure): `dqn.py` (`DQNAgent`), `double_q.py` (`DoubleQAgent`, shared engine), `double_dqn.py` (`DoubleDQNAgent`), `dueling_dqn.py` (`DuelingDQNAgent`), `dqn_atari.py` (`DQNAtariAgent`). For these, the Config/Hypers split is mechanical: move DYNAMIC fields off `self.config` and onto `AgentStateT` (or a new argument to `step`), since `self.config` is exactly the kind of Python-object closure the module docstrings say is fine only for STATIC data.
- **Still `make_train(config, env, env_params) -> train(rng)` closures**: `rainbow.py`, `qrc.py`, `ppo.py`, `sac.py`, `sac_rc.py`, `td3.py`, `greedy_ac.py`. All of these close the *entire* config over the returned `train` function and its nested `_update_step`/`_do_train` closures — there is no `AgentStateT`-shaped carry that could accept a hypers pytree without first restructuring the function into the two-method port. `td3.py` is furthest along informally (network/buffer construction already hoisted out of `train(rng)` into `make_train`'s outer scope), but none of the seven expose a `step`-like per-iteration entry point that a batched-vmap caller could drive with `in_axes=0` today.

## `rl-components` agent Protocol finding

`rl_components.agent_protocol.AgentProtocol` (`libs/rl-components/src/rl_components/agent_protocol.py:94`) is a two-method `Protocol[AgentStateT, ObservationT, ActionT]`: `init(key, spec) -> AgentStateT` and `step(state, timestep, step_index) -> AgentStep[AgentStateT, ActionT]`. Its own docstring (lines 30-35) states explicitly: **"THE AGENT OBJECT ITSELF IS STATIC, NOT A JAX VALUE... The implementation and its configuration are closed over; only `AgentStateT`... moves through the scan carry."** This is precisely the assumption a Config/Hypers split must break: today there is exactly one place config lives (the closed-over Python object), and the Protocol's two method signatures have no slot for a second, dynamic, vmappable argument.

A matching split would need one of:
1. Add a `Hypers`-typed argument to both `init` and `step` (e.g. `step(self, state, hypers, timestep, step_index)`), with the Protocol's generic parameters extended to `AgentProtocol[AgentStateT, HypersT, ObservationT, ActionT]`; or
2. Fold hypers into `AgentStateT` itself as a distinguished sub-pytree, leaving the two-method signature unchanged but requiring every concrete `AgentState` (`DQNAgentState`, `DoubleQAgentState`, `DQNAtariAgentState`, and eventually the closure-style agents' `RunnerState`s once migrated) to carry a `hypers: <AgentName>Hypers` field.

Either way, this is a signature-level change to `AgentProtocol`, not something achievable by only touching `rl-agents` — every current implementer (the five `AgentProtocol`-conformant agents above) would need `step`/`init` updated in lockstep, and the seven closure-style agents would need to gain `AgentProtocol` conformance in the same pass or be left out of the vmap-zone mechanism entirely until migrated.
