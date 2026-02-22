import jax
from rl_agents.ppo import make_train
from rl_components.types import PPOConfig
import argparse
import os

def dump_hlo(agent="ppo"):
    if agent == "ppo":
        config = PPOConfig(TOTAL_TIMESTEPS=100, ENV_NAME="CartPole-v1")
        train_fn = make_train(config)
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
