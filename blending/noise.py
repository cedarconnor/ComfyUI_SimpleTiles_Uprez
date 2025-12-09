"""
Noise-Based Blending

This module implements the NoiseBlender class, which uses Perlin noise
to create organic, irregular boundaries between tiles. This breaks up
the predictable grid patterns of linear blending while maintaining
good performance.

Characteristics:
- Organic, natural-looking seam boundaries
- Breaks up regular grid artifacts
- Good quality/speed balance (~5-10% overhead vs linear)
- Uses Fractional Brownian Motion for multi-scale detail
- Best for general-purpose use

The noise-based approach randomizes the seam position along each edge,
making artifacts much less perceptible to the human eye while maintaining
smooth transitions.
"""

import torch
import math
from .base import TileBlender


class NoiseBlender(TileBlender):
    """
    Perlin noise-based blending for organic tile boundaries.

    Uses 1D Perlin noise with Fractional Brownian Motion (FBM) to create
    naturally varying seam positions. The boundary undulates smoothly along
    each edge, breaking up the linear patterns that create visible artifacts.

    This approach significantly reduces the visibility of tile seams while
    adding minimal computational overhead compared to linear blending.

    Args:
        blend_width: Number of pixels over which blending occurs
        frequency: Base noise frequency (default: 0.05)
                  Lower values = larger, smoother features
                  Higher values = smaller, more chaotic features
        octaves: Number of noise layers to combine (default: 3)
                More octaves = finer detail but slower processing

    Example:
        >>> blender = NoiseBlender(blend_width=48, frequency=0.05, octaves=3)
        >>> mask = blender.create_mask(128, 48, "horizontal", seed=42)
        >>> canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
    """

    def __init__(
        self,
        blend_width: int,
        frequency: float = 0.05,
        octaves: int = 3
    ):
        """
        Initialize noise-based blender.

        Args:
            blend_width: Blending region width in pixels
            frequency: Base noise frequency (default: 0.05)
                      Controls feature size. Typical range: 0.01-0.2
            octaves: Number of noise octaves (default: 3)
                    Typical range: 1-6. More = more detail

        Raises:
            ValueError: If blend_width < 0, frequency <= 0, or octaves < 1
        """
        super().__init__(blend_width)

        if frequency <= 0:
            raise ValueError(f"frequency must be > 0, got {frequency}")
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")

        self.frequency = frequency
        self.octaves = octaves

    def _smoothstep(self, t: torch.Tensor) -> torch.Tensor:
        """
        Smooth interpolation curve using Hermite polynomial.

        Provides smooth acceleration/deceleration for natural-looking
        transitions. The curve has zero derivatives at t=0 and t=1,
        ensuring C1 continuity.

        Formula: 3t² - 2t³

        Args:
            t: Input values, typically in [0, 1]

        Returns:
            Smoothed values with same shape as input

        Example:
            >>> t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
            >>> blender._smoothstep(t)
            tensor([0.0000, 0.1563, 0.5000, 0.8437, 1.0000])
        """
        # Clamp to [0, 1] for safety
        t = torch.clamp(t, 0.0, 1.0)
        # Hermite interpolation: 3t² - 2t³
        return t * t * (3 - 2 * t)

    def _perlin_1d(self, x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """
        Simple 1D Perlin-like noise generator.

        Generates smooth noise by interpolating between random gradients
        at integer positions. This provides coherent noise with continuous
        derivatives, essential for natural-looking boundaries.

        Args:
            x: Input positions (any real values)
            seed: Random seed for reproducibility

        Returns:
            Noise values in approximate range [-1, 1]

        Implementation:
            1. Find integer boundaries around each x position
            2. Generate random gradients at those boundaries (deterministic from seed)
            3. Interpolate gradients using smoothstep
            4. Result is smooth, continuous noise

        Example:
            >>> x = torch.linspace(0, 10, 100)
            >>> noise = blender._perlin_1d(x, seed=42)
            >>> noise.min(), noise.max()  # Approximately [-1, 1]
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # Find integer boundaries
        x0 = x.floor().long()
        x1 = x0 + 1

        # Fractional position within cell
        t = x - x0.float()

        # Apply smoothstep for smooth interpolation
        t = self._smoothstep(t)

        # Generate random gradients at integer positions
        # We need gradients for all positions from min(x0) to max(x1)
        min_idx = max(0, x0.min().item())
        max_idx = x1.max().item() + 1

        # Create gradient table (deterministic from seed)
        gradients = torch.rand(max_idx + 1) * 2 - 1  # Range [-1, 1]

        # Clamp indices to valid range
        x0_clamped = torch.clamp(x0, 0, max_idx)
        x1_clamped = torch.clamp(x1, 0, max_idx)

        # Gather gradients at boundaries
        g0 = gradients[x0_clamped]
        g1 = gradients[x1_clamped]

        # Linear interpolation between gradients
        return g0 * (1 - t) + g1 * t

    def _fbm_1d(self, x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """
        Fractional Brownian Motion: layered multi-scale noise.

        Combines multiple octaves of Perlin noise at different frequencies
        and amplitudes to create natural, detailed patterns. Each octave
        adds finer detail at half the amplitude.

        This technique is commonly used in procedural terrain generation
        and creates more organic, natural-looking patterns than single-
        frequency noise.

        Args:
            x: Input positions
            seed: Base random seed (each octave uses seed + offset)

        Returns:
            Normalized noise values in [0, 1]

        Algorithm:
            result = 0
            amplitude = 1.0
            frequency = 1.0
            for octave in range(octaves):
                result += perlin(x * frequency, seed) * amplitude
                amplitude *= 0.5  (each octave half as strong)
                frequency *= 2.0  (each octave twice as detailed)

        Example:
            >>> x = torch.linspace(0, 10, 200)
            >>> fbm = blender._fbm_1d(x, seed=42)
            >>> fbm.min(), fbm.max()
            (tensor(0.0), tensor(1.0))
        """
        result = torch.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amplitude = 0.0  # Track for normalization

        for i in range(self.octaves):
            # Each octave uses a different seed for variation
            octave_seed = seed + i * 1000

            # Add this octave's contribution
            result += self._perlin_1d(x * frequency, octave_seed) * amplitude

            # Track maximum possible amplitude for normalization
            max_amplitude += amplitude

            # Prepare for next octave
            amplitude *= 0.5  # Halve the amplitude
            frequency *= 2.0  # Double the frequency

        # Normalize to approximately [0, 1]
        # The result is in range [-max_amplitude, max_amplitude]
        result = (result + max_amplitude) / (2 * max_amplitude)

        # Clamp to ensure valid range
        return torch.clamp(result, 0.0, 1.0)

    def create_mask(
        self,
        height: int,
        width: int,
        direction: str,
        seed: int = 0
    ) -> torch.Tensor:
        """
        Create a noise-modulated blend mask.

        Generates a mask where the boundary position varies smoothly
        according to Perlin noise, creating organic, natural-looking
        seams that are much less perceptible than linear gradients.

        The boundary undulates within the central 60% of the blend region
        (30%-70% of width/height) to ensure it stays well away from edges.

        Args:
            height: Mask height in pixels
            width: Mask width in pixels
            direction: Blending direction:
                - "horizontal": Boundary varies along Y, moves in X
                - "vertical": Boundary varies along X, moves in Y
            seed: Random seed for reproducible noise patterns

        Returns:
            Tensor of shape (height, width) with values in [0, 1]
            The boundary position is determined by noise, with smooth
            transitions across the boundary.

        Implementation:
            1. Generate 1D noise along perpendicular axis
            2. Map noise [0,1] to boundary position [0.3, 0.7] of width/height
            3. For each position, compute signed distance from boundary
            4. Apply sigmoid to create smooth S-curve transition

        Example:
            >>> blender = NoiseBlender(48, frequency=0.05, octaves=3)
            >>> mask = blender.create_mask(128, 48, "horizontal", seed=42)
            >>> # Boundary position varies organically along Y axis
        """
        # Softness determines gradient steepness
        # Larger = more gradual transition
        softness = max(width // 4, 8) if direction == "horizontal" else max(height // 4, 8)

        if direction == "horizontal":
            # Noise varies along Y-axis, boundary moves in X-direction
            # This creates a vertical seam with organic horizontal variation

            # Generate noise at each Y position
            y = torch.arange(height).float() * self.frequency
            noise = self._fbm_1d(y, seed)

            # Map noise [0,1] to boundary position [0.3, 0.7] of width
            # This keeps the boundary away from edges
            boundary = noise * 0.4 + 0.3  # [0.3, 0.7]
            boundary = boundary * width  # Scale to pixel positions

            # Create X coordinate grid
            x = torch.arange(width).float().unsqueeze(0).expand(height, -1)
            boundary = boundary.unsqueeze(1).expand(-1, width)

            # Signed distance from boundary (negative=left, positive=right)
            dist = (x - boundary) / softness

            # Sigmoid creates smooth S-curve transition
            # Far left (dist << 0) → 0, Far right (dist >> 0) → 1
            mask = torch.sigmoid(dist)

        elif direction == "vertical":
            # Noise varies along X-axis, boundary moves in Y-direction
            # This creates a horizontal seam with organic vertical variation

            # Generate noise at each X position
            x = torch.arange(width).float() * self.frequency
            noise = self._fbm_1d(x, seed)

            # Map noise to boundary position
            boundary = noise * 0.4 + 0.3  # [0.3, 0.7]
            boundary = boundary * height  # Scale to pixel positions

            # Create Y coordinate grid
            y = torch.arange(height).float().unsqueeze(1).expand(-1, width)
            boundary = boundary.unsqueeze(0).expand(height, -1)

            # Signed distance from boundary
            dist = (y - boundary) / softness

            # Sigmoid transition
            mask = torch.sigmoid(dist)

        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'"
            )

        return mask

    def blend_tiles(
        self,
        canvas: torch.Tensor,
        tile: torch.Tensor,
        position: dict,
        tile_calc: dict
    ) -> torch.Tensor:
        """
        Blend a tile onto canvas using noise-based boundaries.

        Uses the tile's index as a seed for noise generation, ensuring
        each tile gets a unique but reproducible boundary pattern.

        Args:
            canvas: Output canvas (B, H, W, C)
            tile: Tile to blend (B, tile_h, tile_w, C)
            position: Tile placement metadata with keys:
                - "index": Tile index (used as noise seed base)
                - "row", "col": Grid position
                - "place_x", "place_y": Placement coordinates
            tile_calc: Global tiling metadata

        Returns:
            Updated canvas with tile blended in

        Implementation:
            - Non-overlapping center: Direct copy
            - Left edge: Horizontal noise boundary
            - Top edge: Vertical noise boundary
            - Corner: Multiplicative combination of both masks

        Note:
            Different seed offsets for left (base_seed) and top (base_seed+1)
            ensure independent noise patterns for each edge.

        Example:
            >>> blender = NoiseBlender(48, frequency=0.05, octaves=3)
            >>> canvas = torch.zeros(1, 1024, 1024, 3)
            >>> tile = torch.rand(1, 256, 256, 3)
            >>> position = {"index": 5, "row": 1, "col": 2, "place_x": 384, "place_y": 192}
            >>> canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
        """
        blend_w = self.blend_width

        # Extract tile dimensions
        B, th, tw, C = tile.shape

        # Extract placement coordinates
        if "place_x" in position and "place_y" in position:
            px = position["place_x"]
            py = position["place_y"]
        else:
            # Fallback
            px = position.get("x1", 0)
            py = position.get("y1", 0)

        # Determine which edges need blending
        row = position.get("row", 0)
        col = position.get("col", 0)
        has_left = col > 0
        has_top = row > 0

        # Clamp blend width to tile dimensions
        blend_w = min(blend_w, th, tw)

        # Use tile index as seed base for reproducible but varied noise
        idx = position.get("index", 0)
        base_seed = idx * 12345  # Large multiplier ensures well-separated seeds

        # Start with canvas copy
        result = canvas.clone()

        # Calculate non-overlapping center region boundaries
        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0

        # Copy non-overlapping center region directly
        if cy1 < th and cx1 < tw:
            result[:, py+cy1:py+th, px+cx1:px+tw, :] = tile[:, cy1:, cx1:, :]

        # Blend left edge with horizontal noise boundary
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal", seed=base_seed)
            bg = canvas[:, py:py+th, px:px+blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py:py+th, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend top edge with vertical noise boundary
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical", seed=base_seed + 1)
            bg = canvas[:, py:py+blend_w, px:px+tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py:py+blend_w, px:px+tw, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend corner with multiplicative mask combination
        # Multiplicative gives proper corner weighting
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal", seed=base_seed)
            mask_v = self.create_mask(blend_w, blend_w, "vertical", seed=base_seed + 1)

            # Multiply masks to get proper corner behavior
            # Both masks must agree for high weights
            mask_corner = mask_h * mask_v

            bg = canvas[:, py:py+blend_w, px:px+blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py:py+blend_w, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)

        return result

    def __repr__(self) -> str:
        """String representation of NoiseBlender."""
        return (
            f"NoiseBlender(blend_width={self.blend_width}, "
            f"frequency={self.frequency}, octaves={self.octaves})"
        )
