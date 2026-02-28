"""Small tests for experiment-definition bridge — metric whitelist extraction."""

from experiment_definition.bridge import metric_whitelist
from experiment_definition.experiment import Experiment


class TestMetricWhitelist:
    def test_empty_experiment(self):
        exp = Experiment("test")
        wl = metric_whitelist(exp)
        assert wl == frozenset()

    def test_single_metric(self):
        exp = Experiment("test")
        exp.add_metric("reward", type="float", frequency="per_episode")
        wl = metric_whitelist(exp)
        assert wl == frozenset({"reward"})

    def test_multiple_metrics(self):
        exp = Experiment("test")
        exp.add_metric("reward", type="float", frequency="per_episode")
        exp.add_metric("loss", type="float", frequency="per_update")
        exp.add_metric("eval_reward", type="float", frequency="eval_only")
        wl = metric_whitelist(exp)
        assert wl == frozenset({"reward", "loss", "eval_reward"})

    def test_returns_frozenset(self):
        exp = Experiment("test")
        exp.add_metric("x", type="int", frequency="per_episode")
        wl = metric_whitelist(exp)
        assert isinstance(wl, frozenset)
