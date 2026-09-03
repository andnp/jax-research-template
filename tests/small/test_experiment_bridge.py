"""Small tests for experiment-definition bridge — metric whitelist extraction."""

from experiment_definition import Experiment, metric_whitelist


class TestMetricWhitelist:
    def test_empty_experiment(self) -> None:
        exp = Experiment("test")
        wl = metric_whitelist(exp)
        assert wl == frozenset()

    def test_single_metric(self) -> None:
        exp = Experiment("test")
        exp.add_metric("reward", kind="float", frequency="per_episode")
        wl = metric_whitelist(exp)
        assert wl == frozenset({"reward"})

    def test_multiple_metrics(self) -> None:
        exp = Experiment("test")
        exp.add_metric("reward", kind="float", frequency="per_episode")
        exp.add_metric("loss", kind="float", frequency="per_update")
        exp.add_metric("eval_reward", kind="float", frequency="eval_only")
        wl = metric_whitelist(exp)
        assert wl == frozenset({"reward", "loss", "eval_reward"})

    def test_returns_frozenset(self) -> None:
        exp = Experiment("test")
        exp.add_metric("x", kind="int", frequency="per_episode")
        wl = metric_whitelist(exp)
        assert isinstance(wl, frozenset)
