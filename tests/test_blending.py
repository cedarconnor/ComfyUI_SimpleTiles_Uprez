"""
Unit Tests for Tile Blending Module

Tests the blending module components:
- Blender factory (get_blender)
- Mask generation for each blender type
- Blending operations
- Laplacian pyramid operations
- Backwards compatibility

Run with: pytest tests/test_blending.py -v
"""

import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blending import get_blender, TileBlender, LinearBlender, NoiseBlender, LaplacianBlender


class TestBlenderFactory:
    """Tests for the get_blender factory function."""

    def test_get_linear(self):
        """Test factory creates LinearBlender instance."""
        blender = get_blender("linear", 32)
        assert blender is not None
        assert blender.blend_width == 32
        assert isinstance(blender, LinearBlender)

    def test_get_noise(self):
        """Test factory creates NoiseBlender instance."""
        blender = get_blender("noise", 32)
        assert blender is not None
        assert blender.blend_width == 32
        assert isinstance(blender, NoiseBlender)

    def test_get_laplacian(self):
        """Test factory creates LaplacianBlender instance."""
        blender = get_blender("laplacian", 32)
        assert blender is not None
        assert blender.blend_width == 32
        assert isinstance(blender, LaplacianBlender)

    def test_invalid_mode(self):
        """Test factory raises ValueError for invalid mode."""
        with pytest.raises(ValueError, match="Unknown blend mode"):
            get_blender("invalid_mode", 32)

    def test_kwargs_passed_to_noise(self):
        """Test that kwargs are passed to NoiseBlender."""
        blender = get_blender("noise", 32, frequency=0.1, octaves=5)
        assert blender.frequency == 0.1
        assert blender.octaves == 5

    def test_kwargs_passed_to_laplacian(self):
        """Test that kwargs are passed to LaplacianBlender."""
        blender = get_blender("laplacian", 32, levels=6)
        assert blender.levels == 6


class TestMaskGeneration:
    """Tests for mask generation across all blender types."""

    @pytest.mark.parametrize("blender_mode", ["linear", "noise", "laplacian"])
    def test_mask_shape(self, blender_mode):
        """Test that generated masks have correct shape."""
        blender = get_blender(blender_mode, 32)
        mask = blender.create_mask(128, 64, "horizontal")
        assert mask.shape == (128, 64), f"{blender_mode} mask shape incorrect"

    @pytest.mark.parametrize("blender_mode", ["linear", "noise", "laplacian"])
    def test_mask_range(self, blender_mode):
        """Test that mask values are in valid range [0, 1]."""
        blender = get_blender(blender_mode, 32)
        mask = blender.create_mask(128, 64, "horizontal")
        assert mask.min() >= 0.0, f"{blender_mode} mask has values < 0"
        assert mask.max() <= 1.0, f"{blender_mode} mask has values > 1"

    @pytest.mark.parametrize("direction", ["horizontal", "vertical"])
    def test_mask_directions(self, direction):
        """Test mask generation for both directions."""
        blender = get_blender("linear", 32)
        mask = blender.create_mask(128, 64, direction)
        assert mask.shape == (128, 64)

    def test_noise_reproducibility(self):
        """Test that noise masks are reproducible with same seed."""
        blender = get_blender("noise", 32)
        mask1 = blender.create_mask(128, 64, "horizontal", seed=42)
        mask2 = blender.create_mask(128, 64, "horizontal", seed=42)
        assert torch.allclose(mask1, mask2), "Noise masks not reproducible"

    def test_noise_variation(self):
        """Test that noise masks vary with different seeds."""
        blender = get_blender("noise", 32)
        mask1 = blender.create_mask(128, 64, "horizontal", seed=1)
        mask2 = blender.create_mask(128, 64, "horizontal", seed=2)
        assert not torch.allclose(mask1, mask2), "Noise masks not varying"


class TestBlending:
    """Tests for blending operations."""

    @pytest.fixture
    def sample_tiles(self):
        """Create simple test tiles."""
        # Red tile
        tile1 = torch.zeros(1, 64, 64, 3)
        tile1[:, :, :, 0] = 1.0

        # Blue tile
        tile2 = torch.zeros(1, 64, 64, 3)
        tile2[:, :, :, 2] = 1.0

        return tile1, tile2

    @pytest.fixture
    def sample_canvas(self):
        """Create empty canvas."""
        return torch.zeros(1, 128, 128, 3)

    @pytest.mark.parametrize("mode", ["linear", "noise", "laplacian"])
    def test_blend_preserves_shape(self, mode, sample_tiles):
        """Test that blending preserves tensor shapes."""
        tile1, tile2 = sample_tiles
        blender = get_blender(mode, 16)

        mask = blender.create_mask(64, 16, "horizontal")
        result = blender.blend_overlap_region(
            tile1[:, :, :16, :],
            tile2[:, :, :16, :],
            mask
        )

        assert result.shape == (1, 64, 16, 3)

    @pytest.mark.parametrize("mode", ["linear", "noise", "laplacian"])
    def test_blend_interpolates(self, mode, sample_tiles):
        """Test that blending creates smooth interpolation."""
        tile1, tile2 = sample_tiles
        blender = get_blender(mode, 16)

        mask = blender.create_mask(64, 16, "horizontal")
        result = blender.blend_overlap_region(
            tile1[:, :, :16, :],
            tile2[:, :, :16, :],
            mask
        )

        # Result should have mixed values (not pure red or blue)
        mid_value = result[:, :, 8, :].mean()
        assert mid_value > 0.1, f"{mode} not blending (too low)"
        assert mid_value < 0.9, f"{mode} not blending (too high)"

    def test_blend_tiles_integration(self, sample_canvas, sample_tiles):
        """Test full tile blending onto canvas."""
        canvas = sample_canvas
        tile = sample_tiles[0]
        blender = get_blender("linear", 16)

        position = {
            "index": 0,
            "row": 0,
            "col": 0,
            "place_x": 0,
            "place_y": 0,
        }

        tile_calc = {
            "image_height": 128,
            "image_width": 128,
            "overlap": 16,
        }

        result = blender.blend_tiles(canvas, tile, position, tile_calc)
        assert result.shape == canvas.shape
        # First tile should be placed directly
        assert torch.allclose(result[0, :64, :64, :], tile[0])


class TestLaplacianPyramid:
    """Tests specific to Laplacian pyramid operations."""

    def test_pyramid_levels(self):
        """Test that pyramid has correct number of levels."""
        blender = get_blender("laplacian", 32, levels=4)
        img = torch.rand(1, 128, 128, 3)
        pyramid = blender._build_laplacian_pyramid(img)

        assert len(pyramid) == 4
        assert pyramid[0].shape == (1, 128, 128, 3)
        assert pyramid[1].shape == (1, 64, 64, 3)
        assert pyramid[2].shape == (1, 32, 32, 3)
        assert pyramid[3].shape == (1, 16, 16, 3)

    def test_pyramid_reconstruction(self):
        """Test that pyramid can be reconstructed with minimal error."""
        blender = get_blender("laplacian", 32, levels=4)
        img = torch.rand(1, 128, 128, 3)
        pyramid = blender._build_laplacian_pyramid(img)
        reconstructed = blender._collapse_laplacian_pyramid(pyramid)

        # Should reconstruct approximately (some loss due to filtering)
        assert torch.allclose(img, reconstructed, atol=0.01)

    def test_gaussian_kernel(self):
        """Test Gaussian kernel generation."""
        blender = get_blender("laplacian", 32)
        kernel = blender._gaussian_kernel(5, 1.0)

        assert kernel.shape == (5, 5)
        assert torch.isclose(kernel.sum(), torch.tensor(1.0)), "Kernel not normalized"

    def test_downsample_dimensions(self):
        """Test that downsampling halves dimensions."""
        blender = get_blender("laplacian", 32)
        img = torch.rand(1, 128, 128, 3)
        downsampled = blender._downsample(img)

        assert downsampled.shape == (1, 64, 64, 3)

    def test_upsample_dimensions(self):
        """Test that upsampling doubles dimensions."""
        blender = get_blender("laplacian", 32)
        img = torch.rand(1, 64, 64, 3)
        upsampled = blender._upsample(img, (128, 128))

        assert upsampled.shape == (1, 128, 128, 3)


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with existing workflows."""

    @pytest.mark.skip(reason="Integration not yet complete")
    def test_missing_blend_mode_defaults_to_linear(self):
        """Test that missing blend_mode in tile_calc defaults to linear."""
        # This will be tested during integration phase
        pass

    @pytest.mark.skip(reason="Integration not yet complete")
    def test_missing_tile_positions_reconstructed(self):
        """Test that missing tile_positions can be reconstructed."""
        # This will be tested during integration phase
        pass

    @pytest.mark.skip(reason="Integration not yet complete")
    def test_old_workflow_unchanged_output(self):
        """Test that old workflows produce identical output."""
        # This will be tested during integration phase
        pass


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_negative_blend_width(self):
        """Test that negative blend_width raises error."""
        # Can't test factory yet, but can test base class
        with pytest.raises(ValueError, match="blend_width must be >= 0"):
            class DummyBlender(TileBlender):
                def create_mask(self, h, w, d, s=0):
                    return torch.zeros(h, w)
                def blend_tiles(self, c, t, p, tc):
                    return c
            DummyBlender(-1)

    def test_zero_blend_width(self):
        """Test blender with zero blend width."""
        blender = get_blender("linear", 0)
        assert blender.blend_width == 0

    def test_single_channel_image(self):
        """Test blending with grayscale images (C=1)."""
        blender = get_blender("linear", 16)
        tile1 = torch.ones(1, 64, 64, 1) * 0.2
        tile2 = torch.ones(1, 64, 64, 1) * 0.8
        mask = torch.linspace(0, 1, 16).unsqueeze(0).expand(64, 16)

        result = blender.blend_overlap_region(
            tile1[:, :, :16, :],
            tile2[:, :, :16, :],
            mask
        )
        assert result.shape == (1, 64, 16, 1)

    def test_rgba_image(self):
        """Test blending with RGBA images (C=4)."""
        blender = get_blender("linear", 16)
        tile1 = torch.ones(1, 64, 64, 4) * 0.2
        tile2 = torch.ones(1, 64, 64, 4) * 0.8
        mask = torch.linspace(0, 1, 16).unsqueeze(0).expand(64, 16)

        result = blender.blend_overlap_region(
            tile1[:, :, :16, :],
            tile2[:, :, :16, :],
            mask
        )
        assert result.shape == (1, 64, 16, 4)


if __name__ == "__main__":
    # Run tests with: python tests/test_blending.py
    pytest.main([__file__, "-v", "--tb=short"])
