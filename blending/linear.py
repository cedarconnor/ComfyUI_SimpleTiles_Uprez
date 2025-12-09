"""
Linear Gradient Blending

This module implements the LinearBlender class, which provides simple
linear gradient blending for tile merging. This is the current/legacy
blending method used in ComfyUI_SimpleTiles_Uprez.

Characteristics:
- Fast processing with minimal computational overhead
- Simple linear gradients across overlap regions
- Visible grid artifacts, especially in textured or gradient areas
- Best for quick previews or low-detail images

The linear blending approach creates predictable seams due to the
regular gradient patterns, but remains useful for speed-critical
applications.
"""

import torch
from .base import TileBlender


class LinearBlender(TileBlender):
    """
    Simple linear gradient blending implementation.

    Creates smooth transitions using linear gradients across overlap
    regions. While fast, this method produces visible grid artifacts
    at tile boundaries, particularly noticeable in:
    - Textured regions (fabric, foliage, skin)
    - Smooth gradients (skies, backgrounds)
    - High-frequency detail zones

    The blending process:
    1. Non-overlapping tile centers are copied directly
    2. Left edges blend with horizontal gradients
    3. Top edges blend with vertical gradients
    4. Corners use averaged horizontal + vertical gradients

    Example:
        >>> blender = LinearBlender(blend_width=48)
        >>> mask = blender.create_mask(128, 48, "horizontal")
        >>> canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
    """

    def create_mask(
        self,
        height: int,
        width: int,
        direction: str,
        seed: int = 0  # Unused for linear blending
    ) -> torch.Tensor:
        """
        Create a linear gradient mask for blending.

        Generates a simple linear gradient that transitions from 0 to 1
        across the specified dimension. The seed parameter is ignored
        as linear blending is deterministic.

        Args:
            height: Mask height in pixels
            width: Mask width in pixels
            direction: Gradient direction:
                - "horizontal": 0→1 gradient from left to right (vertical seam)
                - "vertical": 0→1 gradient from top to bottom (horizontal seam)
            seed: Ignored for linear blending (kept for API compatibility)

        Returns:
            Tensor of shape (height, width) with linear gradient values [0, 1]

        Example:
            >>> blender = LinearBlender(32)
            >>> mask = blender.create_mask(128, 32, "horizontal")
            >>> mask.shape
            torch.Size([128, 32])
            >>> mask[0, 0], mask[0, -1]  # Left: 0, Right: 1
            (tensor(0.), tensor(1.))
        """
        if direction == "horizontal":
            # Horizontal gradient: blend left-to-right (for vertical seams)
            # Values transition from 0 (left edge) to 1 (right edge)
            mask = torch.linspace(0, 1, width)
            mask = mask.unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            # Vertical gradient: blend top-to-bottom (for horizontal seams)
            # Values transition from 0 (top edge) to 1 (bottom edge)
            mask = torch.linspace(0, 1, height)
            mask = mask.unsqueeze(1).expand(-1, width)
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
        Blend a single tile onto the canvas using linear gradients.

        This method handles the complete tile blending process:
        1. Detects which edges need blending (based on row/col position)
        2. Copies non-overlapping center region directly
        3. Blends left edge if tile has left neighbor
        4. Blends top edge if tile has top neighbor
        5. Blends corner with special handling if both neighbors exist

        The blending matches the original dynamic.py implementation but
        uses a cleaner edge-by-edge approach instead of a single weight matrix.

        Args:
            canvas: Output image tensor (B, H, W, C) being built up
            tile: Single processed tile to blend in (B, tile_h, tile_w, C)
            position: Tile placement metadata dict with keys:
                - "index": Tile index in processing order
                - "row": Grid row (0-indexed)
                - "col": Grid column (0-indexed)
                - "place_x", "place_y": Destination coordinates
            tile_calc: Global tiling metadata dict (overlap, dimensions, etc.)

        Returns:
            Updated canvas tensor with tile blended in

        Notes:
            - First row tiles (row=0) have no top neighbor
            - First column tiles (col=0) have no left neighbor
            - Corners require special blending of both directions
            - Blend width is clamped to tile dimensions to prevent errors

        Example:
            >>> blender = LinearBlender(blend_width=48)
            >>> canvas = torch.zeros(1, 1024, 1024, 3)
            >>> tile = torch.rand(1, 256, 256, 3)
            >>> position = {"row": 1, "col": 1, "place_x": 192, "place_y": 192, ...}
            >>> canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
        """
        blend_w = self.blend_width

        # Extract tile dimensions
        B, th, tw, C = tile.shape

        # Extract placement coordinates
        # Handle both new dict format and legacy tuple reconstruction
        if "place_x" in position and "place_y" in position:
            px = position["place_x"]
            py = position["place_y"]
        else:
            # Fallback: compute from position if needed
            px = position.get("x1", 0)
            py = position.get("y1", 0)

        # Determine which edges need blending
        row = position.get("row", 0)
        col = position.get("col", 0)
        has_left = col > 0
        has_top = row > 0

        # Clamp blend width to tile dimensions
        blend_w = min(blend_w, th, tw)

        # Start with canvas copy
        result = canvas.clone()

        # Calculate non-overlapping center region boundaries
        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0

        # Copy non-overlapping center region directly (no blending needed)
        # This is the "new" content that doesn't overlap with existing tiles
        if cy1 < th and cx1 < tw:
            result[:, py+cy1:py+th, px+cx1:px+tw, :] = tile[:, cy1:, cx1:, :]

        # Blend left edge (vertical seam)
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py:py+th, px:px+blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py:py+th, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend top edge (horizontal seam)
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py:py+blend_w, px:px+tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py:py+blend_w, px:px+tw, :] = self.blend_overlap_region(bg, fg, mask)

        # Blend corner (requires special handling)
        # The corner is blended twice (once horizontal, once vertical)
        # We average the two masks to get proper corner weighting
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")

            # Average the two directions for corner
            # This ensures smooth transition in both dimensions
            mask_corner = (mask_h + mask_v) / 2

            bg = canvas[:, py:py+blend_w, px:px+blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py:py+blend_w, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)

        return result

    def __repr__(self) -> str:
        """String representation of LinearBlender."""
        return f"LinearBlender(blend_width={self.blend_width})"
