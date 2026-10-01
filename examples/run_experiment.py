"""Full experiment lifecycle: definition → training → checkpoint.

Demonstrates how to wire experiment-definition, rl-agents, and research-store
into a cohesive pipeline.

Usage:
    uv run examples/run_experiment.py
"""

import uuid
from pathlib import Path

import gymnax
import jax
import matplotlib.pyplot as plt
from experiment_definition.component import Component, ComponentType
from experiment_definition.experiment import Experiment
from research_store.store import Store
from rl_agents.dqn import DQNAgent, DQNConfig
from rl_components.loop import run
from rl_envs.gymnax_bridge import make_gymnax_env

GAMMA = 0.99


def main() -> None:
    # ── 1. Define the experiment ──────────────────────────────────────────────
    exp = Experiment("DQN CartPole Integration", description="Full pipeline demo")

    dqn_component = Component(
        name="DQN",
        path=Path("domains/rl/agents/src/rl_agents/dqn.py"),
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
    exp.add_metric("episode_return", kind="float")
    exp.add_metric("loss", kind="float")

    # Persist definition to SQLite
    db_path = Path("experiment.sqlite")
    exp.sync(db_path)
    print(f"✓ Experiment definition synced to {db_path}")

    # ── 2. Configure artifact store ───────────────────────────────────────────
    store = Store(
        root=db_path.parent / "artifacts",
        experiment_id="dqn_cartpole_integration",
    )
    artifact_execution_id = uuid.uuid4()
    print(f"✓ Store configured (execution_id={artifact_execution_id})")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    config = DQNConfig(
        env_name="CartPole-v1",
        total_timesteps=100_000,
        seed=42,
    )
    raw_env, env_params = gymnax.make(config.env_name)
    env = make_gymnax_env(raw_env)
    agent = DQNAgent(config)
    def train_one_key(key: jax.Array):
        return run(
            agent,
            env,
            key,
            steps=config.total_timesteps,
            gamma=GAMMA,
            env_params=env_params,
        )

    train_fn = jax.jit(train_one_key)

    print("Training DQN on CartPole-v1 (100k steps)...")
    rng = jax.random.key(config.seed)
    final_state, metrics = train_fn(rng)
    print("✓ Training complete")

    # ── 4. Save checkpoint ────────────────────────────────────────────────────
    final_params = final_state.agent_state.algorithm_state.train_state.params
    uri = store.put(final_params, name="final_weights", execution_id=artifact_execution_id)
    print(f"✓ Checkpoint saved: {uri}")

    # Verify round-trip
    store.get(uri)
    print("✓ Checkpoint recovery verified")

    # ── 5. Plot results ───────────────────────────────────────────────────────
    returns = metrics["loop/episode_return"][metrics["loop/episode_end"]]
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(returns)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title("DQN CartPole — Full Experiment Pipeline")
    fig.savefig("experiment_results.png", dpi=100, bbox_inches="tight")
    print("✓ Results plotted to experiment_results.png")

    print("\n✓ Experiment lifecycle complete!")


if __name__ == "__main__":
    main()
