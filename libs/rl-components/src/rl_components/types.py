from rl_components.structs import chex_struct


@chex_struct(frozen=True)
class PPOConfig:
    # Training Hyperparameters
    LR: float = 2.5e-4
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
    NORMALIZE_OBSERVATIONS: bool = False
    OBS_NORM_EPS: float = 1e-8
    OBS_NORM_CLIP: float = 10.0

    # Environment
    ENV_NAME: str = "MountainCar-v0"
    SEED: int = 42
    DEBUG: bool = False

