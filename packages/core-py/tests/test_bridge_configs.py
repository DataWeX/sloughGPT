"""Tests for domains.collections.training_bridge — TrainingDataConfig; domains.collections.world_bridge — WorldFeedConfig."""

from domains.collections.training_bridge import TrainingDataConfig
from domains.collections.world_bridge import WorldFeedConfig


class TestTrainingDataConfig:
    def test_defaults(self):
        tdc = TrainingDataConfig()
        assert tdc.block_size == 128
        assert tdc.separator == "\n"
        assert tdc.include_metadata is False
        assert tdc.deduplicate is True
        assert tdc.min_length == 10

    def test_custom(self):
        tdc = TrainingDataConfig(block_size=64, separator="|", deduplicate=False)
        assert tdc.block_size == 64
        assert tdc.separator == "|"
        assert tdc.deduplicate is False


class TestWorldFeedConfig:
    def test_defaults(self):
        wfc = WorldFeedConfig()
        assert wfc.grid_size == (64, 32, 64)
        assert wfc.energy_scale == 1.0
        assert wfc.temperature_scale == 1.0
        assert wfc.feed_radius == 5
        assert wfc.max_records == 1000

    def test_custom(self):
        wfc = WorldFeedConfig(grid_size=(32, 16, 32), energy_scale=2.0, feed_radius=10)
        assert wfc.grid_size == (32, 16, 32)
        assert wfc.energy_scale == 2.0
        assert wfc.feed_radius == 10
