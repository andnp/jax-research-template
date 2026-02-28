"""Small tests for rl_components.types — PPOConfig validation."""


from rl_components.types import PPOConfig


class TestPPOConfig:
    def test_defaults(self):
        cfg = PPOConfig()
        assert cfg.LR == 2.5e-4
        assert cfg.GAMMA == 0.99
        assert cfg.GAE_LAMBDA == 0.95
        assert cfg.CLIP_EPS == 0.2
        assert cfg.ENV_NAME == "MountainCar-v0"

    def test_frozen(self):
        cfg = PPOConfig()
        try:
            cfg.LR = 0.1  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass

    def test_custom_values(self):
        cfg = PPOConfig(LR=1e-3, GAMMA=0.95, ENV_NAME="CartPole-v1")
        assert cfg.LR == 1e-3
        assert cfg.GAMMA == 0.95
        assert cfg.ENV_NAME == "CartPole-v1"

    def test_num_updates_computation(self):
        cfg = PPOConfig(TOTAL_TIMESTEPS=1000, NUM_STEPS=100)
        num_updates = cfg.TOTAL_TIMESTEPS // cfg.NUM_STEPS
        assert num_updates == 10

    def test_minibatch_size_computation(self):
        cfg = PPOConfig(NUM_STEPS=128, NUM_MINIBATCHES=4)
        minibatch_size = cfg.NUM_STEPS // cfg.NUM_MINIBATCHES
        assert minibatch_size == 32
