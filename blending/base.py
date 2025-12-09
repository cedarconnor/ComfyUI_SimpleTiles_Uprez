"""
Abstract Base Class for Tile Blending Strategies

This module defines the TileBlender abstract base class that all blending
implementations must inherit from.
"""

from abc import ABC, abstractmethod
import torch


class TileBlender(ABC):
    """
    Abstract base class for tile blending strategies.

    All blending implementations (Linear, Noise, Laplacian) inherit from this
    class and must implement the abstract methods for creating blend masks
    and blending tiles onto a canvas.

    Attributes:
        blend_width (int): Number of pixels over which blending occurs.
                          This defines the width of the overlap region where
                          two tiles are smoothly merged together.
    """

    def __init__(self, blend_width: int):
        """
        Initialize the tile blender.

        Args:
            blend_width: Number of pixels over which blending occurs.
                        Must be <= overlap parameter used in tile splitting.
                        Larger values create smoother transitions but may
                        blur fine details.

        Raises:
            ValueError: If blend_width is negative
        """
        if blend_width < 0:
            raise ValueError(f"blend_width must be >= 0, got {blend_width}")

        self.blend_width = blend_width

    @abstractmethod
    def create_mask(
        self,
        height: int,
        width: int,
        direction: str,
        seed: int = 0
    ) -> torch.Tensor:
        """
        Create a blend mask for the overlap region.

        The mask determines how two overlapping tiles are weighted during
        blending. A mask value of 0 means fully use the background tile,
        while 1 means fully use the foreground tile. Values in between
        create a smooth transition.

        Args:
            height: Height of the overlap region in pixels
            width: Width of the overlap region in pixels
            direction: Blending direction, either:
                      - "horizontal": Blend left-to-right (for vertical seams)
                      - "vertical": Blend top-to-bottom (for horizontal seams)
            seed: Random seed for reproducible noise-based masks (optional).
                 Linear blending ignores this parameter.

        Returns:
            Tensor of shape (height, width) with values in [0, 1].
            - 0 = fully use background tile
            - 1 = fully use foreground tile
            - 0-1 = weighted blend

        Example:
            >>> mask = blender.create_mask(128, 64, "horizontal", seed=42)
            >>> mask.shape
            torch.Size([128, 64])
            >>> mask.min(), mask.max()
            (tensor(0.), tensor(1.))
        """
        pass

    @abstractmethod
    def blend_tiles(
        self,
        canvas: torch.Tensor,
        tile: torch.Tensor,
        position: dict,
        tile_calc: dict
    ) -> torch.Tensor:
        """
        Blend a single tile onto the canvas.

        This is the main blending method that places a processed tile onto
        the output canvas, handling overlaps with already-placed tiles by
        creating appropriate blend masks for each edge.

        Args:
            canvas: Output canvas tensor of shape (B, H, W, C) where:
                   - B = batch size (typically 1)
                   - H = full output image height
                   - W = full output image width
                   - C = number of channels (3 for RGB, 4 for RGBA)

            tile: Single tile tensor of shape (B, tile_h, tile_w, C)
                 This is the processed/upscaled tile to be placed

            position: Dictionary containing tile placement metadata:
                - "index" (int): Tile index in processing order
                - "row" (int): Grid row position (0-indexed)
                - "col" (int): Grid column position (0-indexed)
                - "x1", "y1" (int): Top-left crop coordinates (source)
                - "x2", "y2" (int): Bottom-right crop coordinates (source)
                - "place_x", "place_y" (int): Destination placement coordinates

            tile_calc: Dictionary containing global tiling metadata:
                - "image_height", "image_width" (int): Full output size
                - "tile_height", "tile_width" (int): Base tile dimensions
                - "overlap", "overlap_x", "overlap_y" (int): Overlap sizes
                - "rows", "cols" (int): Grid dimensions
                - "blend_mode" (str): Current blend mode
                - "tile_positions" (list): Metadata for all tiles

        Returns:
            Updated canvas tensor with the tile blended in, same shape as input
            canvas: (B, H, W, C)

        Implementation Notes:
            - Detect which edges need blending (left, top, corner)
            - First-row and first-column tiles have no top/left neighbors
            - Use create_mask() to generate appropriate blend masks
            - Handle corners specially (combination of horizontal + vertical)
            - Non-overlapping regions can be directly copied without blending

        Example:
            >>> canvas = torch.zeros(1, 1024, 1024, 3)
            >>> tile = torch.rand(1, 256, 256, 3)
            >>> position = {"row": 1, "col": 1, "place_x": 192, "place_y": 192, ...}
            >>> canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
        """
        pass

    def blend_overlap_region(
        self,
        background: torch.Tensor,
        foreground: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply mask-based blending to overlapping regions.

        This is a shared helper method that performs the actual weighted
        blending operation given a background tile, foreground tile, and
        blend mask.

        Formula: result = background * (1 - mask) + foreground * mask

        Args:
            background: Background tile tensor of shape (B, H, W, C)
                       This is the already-placed tile content

            foreground: Foreground tile tensor of shape (B, H, W, C)
                       This is the new tile being blended in
                       Same dimensions as background

            mask: Blend mask tensor of shape (H, W)
                 Values in [0, 1] determining blend weights
                 Will be automatically expanded to match background/foreground

        Returns:
            Blended result tensor of shape (B, H, W, C)

        Example:
            >>> bg = torch.ones(1, 64, 64, 3) * 0.2
            >>> fg = torch.ones(1, 64, 64, 3) * 0.8
            >>> mask = torch.linspace(0, 1, 64).unsqueeze(0).expand(64, 64)
            >>> result = blender.blend_overlap_region(bg, fg, mask)
            >>> result[0, 0, 0, 0]  # Left edge: mostly background
            tensor(0.2...)
            >>> result[0, 0, -1, 0]  # Right edge: mostly foreground
            tensor(0.8...)
        """
        # Expand mask dimensions: (H, W) -> (1, H, W, 1)
        # This allows broadcasting across batch and channel dimensions
        mask_expanded = mask.unsqueeze(0).unsqueeze(-1)

        # Linear interpolation between background and foreground
        # When mask=0: result = background * 1 + foreground * 0 = background
        # When mask=1: result = background * 0 + foreground * 1 = foreground
        return background * (1 - mask_expanded) + foreground * mask_expanded

    def __repr__(self) -> str:
        """String representation of the blender."""
        return f"{self.__class__.__name__}(blend_width={self.blend_width})"
