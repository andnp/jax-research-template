import chex


@chex.dataclass(frozen=True)
class PPOConfig:
    # Training Hyperparameters
    LR: float = 2.5e-4
    NUM_ENVS: int = 4
    NUM_STEPS: int = 128
    TOTAL_TIMESTEPS: int = 500_000
    UPDATE_EPOCHS: int = 4
    NUM_MINIBATCHES: int = 4

    # PPO Specifics
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5

    # Environment
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42
    DEBUG: bool = False
