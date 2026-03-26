"""Tests for the tiling algorithm."""

import numpy as np
import pytest

from spatialprot_data.utils.helpers.crop import best_mask_tiling_try_to_stop, Tile


class TestFullMask:
    def test_covers_everything(self):
        mask = np.ones((256, 256), dtype=np.uint8)
        tiles, stats, covered = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=64, tolerance=0.0, min_gain_ratio=0.0
        )
        assert stats["coverage_ratio"] == 1.0
        assert stats["num_tiles"] == 16

    def test_tile_coordinates_within_bounds(self):
        mask = np.ones((300, 400), dtype=np.uint8)
        tiles, stats, covered = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=32, tolerance=0.0, min_gain_ratio=0.0
        )
        H, W = covered.shape
        for t in tiles:
            assert 0 <= t.y and t.y + t.h <= H
            assert 0 <= t.x and t.x + t.w <= W


class TestEmptyMask:
    def test_returns_no_tiles(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        tiles, stats, covered = best_mask_tiling_try_to_stop(mask, tile_size=64)
        assert len(tiles) == 0
        assert stats["stop_reason"] == "empty_mask"
        assert stats["coverage_ratio"] == 1.0


class TestPartialMask:
    def test_respects_tolerance(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[:128, :128] = 1
        tiles, stats, covered = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=64, tolerance=0.0, min_gain_ratio=0.0
        )
        for t in tiles:
            assert t.valid_ratio >= 1.0

    def test_higher_tolerance_more_tiles(self):
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[:128, :128] = 1
        mask[130:140, 130:140] = 1  # small island
        _, stats_strict, _ = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=64, tolerance=0.0, min_gain_ratio=0.0
        )
        _, stats_loose, _ = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=64, tolerance=0.8, min_gain_ratio=0.0
        )
        assert stats_loose["num_tiles"] >= stats_strict["num_tiles"]


class TestNoOverlap:
    def test_no_tiles_overlap(self):
        mask = np.ones((256, 256), dtype=np.uint8)
        tiles, _, _ = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=32, allow_overlap=False, min_gain_ratio=0.0
        )
        for i, a in enumerate(tiles):
            for b in tiles[i + 1:]:
                x_overlap = (a.x < b.x + b.w) and (b.x < a.x + a.w)
                y_overlap = (a.y < b.y + b.h) and (b.y < a.y + a.h)
                assert not (x_overlap and y_overlap), f"Tiles {a} and {b} overlap"


class TestMaxTiles:
    def test_max_tiles_respected(self):
        mask = np.ones((256, 256), dtype=np.uint8)
        tiles, stats, _ = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=64, max_tiles=3, min_gain_ratio=0.0
        )
        assert len(tiles) <= 3
        assert stats["stop_reason"] == "max_tiles"


class TestAdaptiveStop:
    def test_coverage_goal_early_stop(self):
        mask = np.ones((256, 256), dtype=np.uint8)
        tiles, stats, _ = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=64, coverage_goal=0.5, min_gain_ratio=0.0
        )
        assert stats["coverage_ratio"] >= 0.5

    def test_adaptive_stop_sparse_mask(self):
        # Sparse mask where gain drops quickly
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[100:200, 100:200] = 1
        mask[300:310, 300:310] = 1  # tiny island
        tiles, stats, _ = best_mask_tiling_try_to_stop(
            mask, tile_size=64, stride=32, coverage_goal=0.8, min_gain_ratio=0.5
        )
        assert stats["stop_reason"] in ("adaptive_stop", "candidates_exhausted", "no_gain")


class TestSmallMask:
    def test_smaller_than_tile_padded(self):
        mask = np.ones((32, 32), dtype=np.uint8)
        tiles, stats, covered = best_mask_tiling_try_to_stop(mask, tile_size=64, tolerance=0.8)
        assert len(tiles) >= 1
        assert covered.shape[0] >= 64 and covered.shape[1] >= 64


class TestTileDataclass:
    def test_fields(self):
        mask = np.ones((128, 128), dtype=np.uint8)
        tiles, _, _ = best_mask_tiling_try_to_stop(mask, tile_size=64, stride=64, min_gain_ratio=0.0)
        assert len(tiles) > 0
        t = tiles[0]
        assert hasattr(t, "y")
        assert hasattr(t, "x")
        assert hasattr(t, "h")
        assert hasattr(t, "w")
        assert hasattr(t, "valid_ratio")
        assert hasattr(t, "gain")

    def test_covered_mask_shape(self):
        mask = np.ones((200, 300), dtype=np.uint8)
        _, _, covered = best_mask_tiling_try_to_stop(mask, tile_size=64, stride=64, min_gain_ratio=0.0)
        # covered_mask shape should match the (possibly padded) mask
        assert covered.shape[0] >= 200
        assert covered.shape[1] >= 300
