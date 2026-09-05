import argparse
import os

import gymnax
import jax
from rl_agents.ppo import PPOAgent
from rl_components.gymnax_bridge import make_gymnax_env
from rl_components.loop import run
from rl_components.types import PPOConfig

GAMMA = 0.99


def dump_hlo(agent: str = "ppo") -> None:
    if agent == "ppo":
        config = PPOConfig(TOTAL_TIMESTEPS=100, ENV_NAME="CartPole-v1")
        raw_env, env_params = gymnax.make(config.ENV_NAME)
        env = make_gymnax_env(raw_env)
        ppo_agent = PPOAgent(config)
        def train_fn(key: jax.Array):
            return run(
                ppo_agent, env, key, steps=config.TOTAL_TIMESTEPS, gamma=GAMMA, env_params=env_params
            )
    else:
        raise ValueError(f"Agent {agent} not supported")

    rng = jax.random.PRNGKey(42)

    # Get the lowered representation
    lowered = jax.jit(train_fn).lower(rng)

    # Get HLO as text
    hlo_text = lowered.as_text()

    os.makedirs("hlo_dumps", exist_ok=True)
    with open(f"hlo_dumps/{agent}_hlo.txt", "w") as f:
        f.write(hlo_text)

    print(f"HLO text dumped to hlo_dumps/{agent}_hlo.txt")

    # You can also get the HLO proto if needed for more advanced tools
    # hlo_proto = lowered.compile().hlo_modules()[0].to_proto()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="ppo")
    args = parser.parse_args()
    dump_hlo(args.agent)
