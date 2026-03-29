"""Full experiment lifecycle: definition → training → metrics → checkpoint.

Demonstrates how to wire experiment-definition, rl-agents,
research-instrument, and research-store into a cohesive pipeline.

Usage:
    uv run examples/run_experiment.py
"""

import uuid
from pathlib import Path

import gymnax
import gymnax.wrappers
import jax
import matplotlib.pyplot as plt
from experiment_definition.bridge import metric_whitelist
from experiment_definition.component import Component, ComponentType
from experiment_definition.experiment import Experiment
from research_instrument.collector import configure
from research_instrument.sqlite_backend import SQLiteBackend
from research_store.store import Store
from rl_agents.dqn import DQNConfig, make_train


def main():
    # ── 1. Define the experiment ──────────────────────────────────────────────
    exp = Experiment("DQN CartPole Integration", description="Full pipeline demo")

    dqn_component = Component(
        name="DQN",
        path=Path("libs/rl-agents/src/rl_agents/dqn.py"),
        type=ComponentType.ALGO,
    )
    env_component = Component(
        name="CartPole-v1",
        path=Path("libs/rl-components/src/rl_components/buffers.py"),
        type=ComponentType.ENV,
    )

    with exp.for_component(dqn_component):
        exp.add_parameter("lr", [3e-4])
        exp.add_parameter("gamma", [0.99])

    with exp.for_component(env_component):
        exp.add_parameter("env_name", ["CartPole-v1"])

    exp.add_parameter("seed", list(range(3)))
    exp.add_metric("returned_episode_returns", type="float", frequency="per_episode")
    exp.add_metric("loss", type="float", frequency="per_update")

    # Persist definition to SQLite
    db_path = Path("experiment.sqlite")
    exp.sync(db_path)
    print(f"✓ Experiment definition synced to {db_path}")

    # ── 2. Configure metrics collection ───────────────────────────────────────
    whitelist = metric_whitelist(exp)
    print(f"✓ Metric whitelist: {whitelist}")

    metrics_db = Path("metrics.db")
    backend = SQLiteBackend(metrics_db, batch_size=50)
    configure(whitelist, backend=backend)
    print(f"✓ Metrics backend: {metrics_db}")

    # ── 3. Configure artifact store ───────────────────────────────────────────
    store = Store(experiment_id="dqn_cartpole_integration")
    execution_id = uuid.uuid4()
    print(f"✓ Store configured (execution_id={execution_id})")

    # ── 4. Train ──────────────────────────────────────────────────────────────
    config = DQNConfig(
        ENV_NAME="CartPole-v1",
        TOTAL_TIMESTEPS=100_000,
        SEED=42,
    )
    env, env_params = gymnax.make(config.ENV_NAME)
    env = gymnax.wrappers.LogWrapper(env)
    train_fn = make_train(config, env=env, env_params=env_params)
    train_fn = jax.jit(train_fn)

    print("Training DQN on CartPole-v1 (100k steps)...")
    rng = jax.random.key(config.SEED)
    result = train_fn(rng)
    print("✓ Training complete")

    # ── 5. Save checkpoint ────────────────────────────────────────────────────
    runner_state = result["runner_state"]
    final_params = runner_state[0].params
    uri = store.put(final_params, name="final_weights", execution_id=execution_id)
    print(f"✓ Checkpoint saved: {uri}")

    # Verify round-trip
    store.get(uri)
    print("✓ Checkpoint recovery verified")

    # ── 6. Flush metrics and report ───────────────────────────────────────────
    backend.flush()
    names = backend.metric_names()
    print(f"✓ Metrics collected: {names}")

    # ── 7. Plot results ───────────────────────────────────────────────────────
    returns = result["metrics"]["returned_episode_returns"]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(returns)
    ax.set_xlabel("Step")
    ax.set_ylabel("Episode Return")
    ax.set_title("DQN CartPole — Full Experiment Pipeline")
    fig.savefig("experiment_results.png", dpi=100, bbox_inches="tight")
    print("✓ Results plotted to experiment_results.png")

    backend.close()
    print("\n✓ Experiment lifecycle complete!")


if __name__ == "__main__":
    main()
