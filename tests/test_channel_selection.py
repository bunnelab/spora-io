"""Tests for channel selection transforms."""

import numpy as np
import pytest

from spatialprot_data.utils.dataset.channels import (
    BaseChannelSelector,
    DropChannelsFraction,
    DropChannelsFixedNumber,
    DropChannelsFixedNumberRange,
    DropChannelsNuclearKnown,
    HierarchicalChannelSampling,
)


class TestDropChannelsFraction:
    def test_output_size(self):
        t = DropChannelsFraction(p=1.0, fraction_range=(0.5, 0.5), rng=np.random.default_rng(0))
        x = np.arange(10)[:, None]
        (x2,) = t(x)
        assert len(x2) == int(np.ceil(0.5 * 10))

    def test_p_zero_noop(self):
        t = DropChannelsFraction(p=0.0, fraction_range=(0.5, 0.5), rng=np.random.default_rng(0))
        x = np.arange(10)[:, None]
        result = t(x)
        # When not applied, returns original args tuple
        assert isinstance(result, tuple)
        assert np.array_equal(result[0], x)

    def test_fraction_one_keeps_all(self):
        t = DropChannelsFraction(p=1.0, fraction_range=(1.0, 1.0), rng=np.random.default_rng(0))
        x = np.arange(8)[:, None]
        (x2,) = t(x)
        assert len(x2) == 8

    def test_invalid_fraction_range_raises(self):
        with pytest.raises(ValueError):
            DropChannelsFraction(p=1.0, fraction_range=(0.8, 0.3))


class TestDropChannelsFixedNumber:
    def test_keeps_exact_number(self):
        t = DropChannelsFixedNumber(p=1.0, num_keep=3, rng=np.random.default_rng(0))
        x = np.arange(8)[:, None]
        (x2,) = t(x)
        assert len(x2) == 3

    def test_clamps_to_available(self):
        t = DropChannelsFixedNumber(p=1.0, num_keep=20, rng=np.random.default_rng(0))
        x = np.arange(5)[:, None]
        (x2,) = t(x)
        assert len(x2) == 5

    def test_negative_num_keep_raises(self):
        with pytest.raises(ValueError):
            DropChannelsFixedNumber(p=1.0, num_keep=-1)


class TestDropChannelsFixedNumberRange:
    def test_within_range(self):
        rng = np.random.default_rng(42)
        t = DropChannelsFixedNumberRange(p=1.0, num_keep_min=2, num_keep_max=5, rng=rng)
        x = np.arange(10)[:, None]
        results = [len(t(x)[0]) for _ in range(100)]
        assert all(2 <= r <= 5 for r in results)

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError):
            DropChannelsFixedNumberRange(p=1.0, num_keep_min=5, num_keep_max=2)


class TestDropChannelsNuclearKnown:
    def test_always_includes_fixed(self):
        rng = np.random.default_rng(0)
        t = DropChannelsNuclearKnown(num_choose=2, p=1.0, rng=rng)
        x = np.arange(6)
        for _ in range(50):
            (x2,) = t(x, fixed_index=0)
            assert 0 in x2

    def test_minus_one_keeps_all(self):
        t = DropChannelsNuclearKnown(num_choose=-1, p=1.0, rng=np.random.default_rng(0))
        x = np.arange(6)
        (x2,) = t(x, fixed_index=0)
        assert len(x2) == 6

    def test_missing_fixed_index_raises(self):
        t = DropChannelsNuclearKnown(num_choose=2, p=1.0)
        x = np.arange(6)
        with pytest.raises(ValueError):
            t(x)


class TestHierarchicalChannelSampling:
    def test_min_channels_respected(self):
        rng = np.random.default_rng(42)
        t = HierarchicalChannelSampling(min_channels=3, p=1.0, rng=rng)
        x = np.arange(10)[:, None]
        results = [len(t(x)[0]) for _ in range(100)]
        assert all(r >= 3 for r in results)

    def test_max_is_n(self):
        rng = np.random.default_rng(42)
        t = HierarchicalChannelSampling(min_channels=1, p=1.0, rng=rng)
        x = np.arange(5)[:, None]
        results = [len(t(x)[0]) for _ in range(200)]
        assert max(results) == 5


class TestBaseChannelSelector:
    def test_invalid_p_raises(self):
        with pytest.raises(ValueError):
            DropChannelsFraction(p=1.5, fraction_range=(0.5, 0.5))

    def test_multiple_inputs_sliced_consistently(self):
        t = DropChannelsFixedNumber(p=1.0, num_keep=3, rng=np.random.default_rng(0))
        x = np.arange(8)[:, None]
        y = np.arange(8)[:, None] * 10
        x2, y2 = t(x, y)
        assert len(x2) == len(y2) == 3
        # Same indices selected for both
        assert np.array_equal(x2 * 10, y2)

    def test_reproducibility_with_same_seed(self):
        x = np.arange(10)[:, None]
        t1 = DropChannelsFraction(p=1.0, fraction_range=(0.5, 0.5), rng=np.random.default_rng(99))
        t2 = DropChannelsFraction(p=1.0, fraction_range=(0.5, 0.5), rng=np.random.default_rng(99))
        (r1,) = t1(x)
        (r2,) = t2(x)
        assert np.array_equal(r1, r2)
