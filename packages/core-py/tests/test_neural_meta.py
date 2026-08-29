"""Tests for domains.soul.cognitive — NeuralPlasticityEngine, MetaLearningEngine."""

from domains.soul.cognitive import NeuralPlasticityEngine, MetaLearningEngine


class TestNeuralPlasticityEngine:
    def test_init(self):
        npe = NeuralPlasticityEngine()
        assert npe.learning_rate == 0.01
        assert len(npe.connections) == 0

    def test_activate(self):
        npe = NeuralPlasticityEngine()
        npe.activate("n1", 0.8)
        assert len(npe.activation_history["n1"]) == 1

    def test_hebbian_learn(self):
        npe = NeuralPlasticityEngine()
        npe.activate("pre", 1.0)
        npe.activate("post", 1.0)
        weight = npe.hebbian_learn("pre", "post")
        assert weight > 0.0

    def test_get_connection_strength(self):
        npe = NeuralPlasticityEngine()
        assert npe.get_connection_strength("a", "b") == 0.0


class TestMetaLearningEngine:
    def test_init(self):
        mle = MetaLearningEngine()
        assert len(mle.strategies) == 4
        assert mle.best_strategy == "spaced"

    def test_record_outcome(self):
        mle = MetaLearningEngine()
        mle.record_outcome("rote", True)
        mle.record_outcome("rote", False)
        assert mle.strategies["rote"]["attempts"] == 2
        assert mle.strategies["rote"]["success"] == 1

    def test_update_weights(self):
        mle = MetaLearningEngine()
        for _ in range(10):
            mle.record_outcome("rote", True)
        mle.update_weights()
        assert mle.strategies["rote"]["weight"] == 1.0
        assert mle.strategies["rote"]["success"] == 10

    def test_get_strategy(self):
        mle = MetaLearningEngine()
        s = mle.get_strategy()
        assert s in mle.strategies
