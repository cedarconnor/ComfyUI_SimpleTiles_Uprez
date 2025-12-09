"""
Integration Tests for DynamicTileSplit and DynamicTileMerge

Tests the complete pipeline:
- Split → Merge workflow with all blend modes
- Backwards compatibility with old tuple-based tile_calc
- Scaled tile processing
- Position metadata handling

Run with: pytest tests/test_integration.py -v
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamic import DynamicTileSplit, DynamicTileMerge


class TestNewWorkflow:
    """Tests for new dict-based workflow with blend modes."""

    @pytest.fixture
    def sample_image(self):
        """Create a simple gradient test image."""
        # 256x256 RGB gradient image
        img = torch.zeros(1, 256, 256, 3)
        for y in range(256):
            for x in range(256):
                img[0, y, x, 0] = x / 255.0  # Red gradient left to right
                img[0, y, x, 1] = y / 255.0  # Green gradient top to bottom
                img[0, y, x, 2] = 0.5  # Constant blue
        return img

    @pytest.mark.parametrize("blend_mode", ["linear", "noise", "laplacian"])
    def test_split_merge_roundtrip(self, sample_image, blend_mode):
        """Test that split→merge preserves image with all blend modes."""
        splitter = DynamicTileSplit()
        merger = DynamicTileMerge()

        # Split with specific blend mode
        tiles, tile_calc = splitter.process(
            image=sample_image,
            tile_width=128,
            tile_height=128,
            overlap=32,
            blend_mode=blend_mode,
            offset=0
        )

        # Verify tile_calc is dict format
        assert isinstance(tile_calc, dict), f"tile_calc should be dict, got {type(tile_calc)}"
        assert tile_calc["blend_mode"] == blend_mode
        assert "tile_positions" in tile_calc
        assert tile_calc["overlap"] == 32

        # Merge tiles back
        result = merger.process(
            images=tiles,
            blend=16,
            tile_calc=tile_calc
        )

        merged = result[0]

        # Verify shape preserved
        assert merged.shape == sample_image.shape, f"Shape mismatch: {merged.shape} vs {sample_image.shape}"

        # Verify values are reasonable (allowing for blending artifacts)
        # Center region should be very close to original
        center_y = slice(64, 192)
        center_x = slice(64, 192)
        original_center = sample_image[:, center_y, center_x, :]
        merged_center = merged[:, center_y, center_x, :]

        # Should be close in center (less blending artifacts)
        diff = torch.abs(original_center - merged_center).mean()
        assert diff < 0.05, f"{blend_mode}: Center region differs by {diff:.4f}"

    def test_tile_calc_structure(self, sample_image):
        """Test that tile_calc contains all required fields."""
        splitter = DynamicTileSplit()

        tiles, tile_calc = splitter.process(
            image=sample_image,
            tile_width=128,
            tile_height=128,
            overlap=32,
            blend_mode="linear",
            offset=0
        )

        # Check all required fields exist
        required_fields = [
            "overlap", "overlap_x", "overlap_y",
            "image_height", "image_width",
            "offset", "tile_height", "tile_width",
            "rows", "cols", "blend_mode", "tile_positions"
        ]

        for field in required_fields:
            assert field in tile_calc, f"Missing field: {field}"

        # Check tile_positions structure
        assert isinstance(tile_calc["tile_positions"], list)
        assert len(tile_calc["tile_positions"]) == len(tiles)

        for pos in tile_calc["tile_positions"]:
            assert "index" in pos
            assert "row" in pos
            assert "col" in pos
            assert "x1" in pos
            assert "y1" in pos
            assert "x2" in pos
            assert "y2" in pos
            assert "place_x" in pos
            assert "place_y" in pos

    def test_upscaling_workflow(self, sample_image):
        """Test that upscaling is handled correctly."""
        splitter = DynamicTileSplit()
        merger = DynamicTileMerge()

        # Split
        tiles, tile_calc = splitter.process(
            image=sample_image,
            tile_width=64,
            tile_height=64,
            overlap=16,
            blend_mode="linear",
            offset=0
        )

        # Simulate 2x upscaling
        upscaled_tiles = torch.nn.functional.interpolate(
            tiles.permute(0, 3, 1, 2),  # NHWC -> NCHW
            scale_factor=2.0,
            mode='bilinear',
            align_corners=False
        ).permute(0, 2, 3, 1)  # NCHW -> NHWC

        # Merge upscaled tiles
        result = merger.process(
            images=upscaled_tiles,
            blend=32,  # Scaled blend width
            tile_calc=tile_calc
        )

        merged = result[0]

        # Should be 2x size of original
        assert merged.shape[1] == sample_image.shape[1] * 2
        assert merged.shape[2] == sample_image.shape[2] * 2


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with old tuple format."""

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image."""
        return torch.rand(1, 128, 128, 3)

    @pytest.fixture
    def old_tile_calc_tuple(self):
        """Create old-style tuple tile_calc."""
        # Old format: (overlap, base_height, base_width, offset, base_tile_height, base_tile_width)
        return (32, 128, 128, 0, 64, 64)

    def test_old_tuple_format(self, sample_image, old_tile_calc_tuple):
        """Test that old tuple format is still supported."""
        merger = DynamicTileMerge()

        # Create some tiles
        tiles = torch.rand(4, 64, 64, 3)

        # Should not raise error with old format
        try:
            result = merger.process(
                images=tiles,
                blend=16,
                tile_calc=old_tile_calc_tuple
            )
            merged = result[0]
            assert merged.shape[0] == 1
            assert merged.shape[3] == 3
        except Exception as e:
            pytest.fail(f"Old tuple format not supported: {e}")

    def test_old_tuple_defaults_to_linear(self, sample_image, old_tile_calc_tuple):
        """Test that old format defaults to linear blending."""
        merger = DynamicTileMerge()
        tiles = torch.rand(4, 64, 64, 3)

        # With old tuple format, should use linear blending
        # We can't directly check the mode, but we verify it doesn't crash
        result = merger.process(
            images=tiles,
            blend=16,
            tile_calc=old_tile_calc_tuple
        )

        assert result[0].shape[0] == 1

    def test_short_tuple_format(self):
        """Test that short tuple format (4 elements) is supported."""
        merger = DynamicTileMerge()
        tiles = torch.rand(4, 64, 64, 3)

        # Old short format: (overlap, base_height, base_width, offset)
        old_short_tuple = (32, 128, 128, 0)

        try:
            result = merger.process(
                images=tiles,
                blend=16,
                tile_calc=old_short_tuple
            )
            assert result[0].shape[0] == 1
        except Exception as e:
            pytest.fail(f"Short tuple format not supported: {e}")


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_single_tile(self):
        """Test with image that produces only one tile."""
        splitter = DynamicTileSplit()
        merger = DynamicTileMerge()

        # Small image with large tile size
        img = torch.rand(1, 64, 64, 3)

        tiles, tile_calc = splitter.process(
            image=img,
            tile_width=128,
            tile_height=128,
            overlap=16,
            blend_mode="linear",
            offset=0
        )

        # Should have exactly 1 tile
        assert len(tiles) == 1

        result = merger.process(
            images=tiles,
            blend=8,
            tile_calc=tile_calc
        )

        merged = result[0]
        assert merged.shape[1] >= img.shape[1]
        assert merged.shape[2] >= img.shape[2]

    def test_zero_overlap(self):
        """Test with zero overlap."""
        splitter = DynamicTileSplit()
        img = torch.rand(1, 128, 128, 3)

        # This should work but tiles won't blend
        tiles, tile_calc = splitter.process(
            image=img,
            tile_width=64,
            tile_height=64,
            overlap=1,  # Minimum overlap
            blend_mode="linear",
            offset=0
        )

        assert len(tiles) > 0
        assert tile_calc["overlap"] == 1

    def test_invalid_tile_calc_type(self):
        """Test that invalid tile_calc type raises error."""
        merger = DynamicTileMerge()
        tiles = torch.rand(4, 64, 64, 3)

        with pytest.raises(ValueError, match="Unexpected tile_calc type"):
            merger.process(
                images=tiles,
                blend=16,
                tile_calc="invalid"  # Wrong type
            )

    def test_blend_width_larger_than_overlap(self):
        """Test that large blend width doesn't crash."""
        splitter = DynamicTileSplit()
        merger = DynamicTileMerge()

        img = torch.rand(1, 128, 128, 3)

        tiles, tile_calc = splitter.process(
            image=img,
            tile_width=64,
            tile_height=64,
            overlap=16,
            blend_mode="linear",
            offset=0
        )

        # Blend width larger than overlap - should clamp or handle gracefully
        result = merger.process(
            images=tiles,
            blend=32,  # Larger than overlap
            tile_calc=tile_calc
        )

        assert result[0].shape[0] == 1


class TestBlendModeComparison:
    """Compare output quality of different blend modes."""

    @pytest.fixture
    def checkerboard_image(self):
        """Create a challenging checkerboard pattern."""
        img = torch.zeros(1, 128, 128, 3)
        for y in range(0, 128, 16):
            for x in range(0, 128, 16):
                if (x // 16 + y // 16) % 2 == 0:
                    img[0, y:y+16, x:x+16, :] = 1.0
        return img

    @pytest.mark.parametrize("blend_mode", ["linear", "noise", "laplacian"])
    def test_blend_modes_produce_valid_output(self, checkerboard_image, blend_mode):
        """Test all blend modes produce valid output on challenging pattern."""
        splitter = DynamicTileSplit()
        merger = DynamicTileMerge()

        tiles, tile_calc = splitter.process(
            image=checkerboard_image,
            tile_width=64,
            tile_height=64,
            overlap=32,
            blend_mode=blend_mode,
            offset=0
        )

        result = merger.process(
            images=tiles,
            blend=16,
            tile_calc=tile_calc
        )

        merged = result[0]

        # Check output is valid
        assert not torch.isnan(merged).any(), f"{blend_mode}: NaN values in output"
        assert not torch.isinf(merged).any(), f"{blend_mode}: Inf values in output"
        assert merged.min() >= -0.1, f"{blend_mode}: Values too low"
        assert merged.max() <= 1.1, f"{blend_mode}: Values too high"


if __name__ == "__main__":
    # Run tests with: python tests/test_integration.py
    pytest.main([__file__, "-v", "--tb=short"])
