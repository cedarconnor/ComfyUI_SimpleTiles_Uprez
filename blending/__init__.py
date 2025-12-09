"""
Advanced Tile Blending Module for ComfyUI_SimpleTiles_Uprez

This module provides multiple blending strategies for seamless tile merging:
- LinearBlender: Simple gradient blending (fast, visible artifacts)
- NoiseBlender: Perlin noise boundaries (balanced quality/speed)
- LaplacianBlender: Multi-band pyramid blending (maximum quality)

Usage:
    from blending import get_blender

    blender = get_blender("noise", blend_width=48)
    canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
"""

from .base import TileBlender
from .linear import LinearBlender
from .noise import NoiseBlender
from .laplacian import LaplacianBlender

BLENDERS = {
    "linear": LinearBlender,
    "noise": NoiseBlender,
    "laplacian": LaplacianBlender,
}

def get_blender(mode: str, blend_width: int, **kwargs):
    """
    Factory function to create appropriate tile blender.

    Args:
        mode: One of "linear", "noise", "laplacian"
        blend_width: Number of pixels over which blending occurs.
                    Must be <= overlap.
        **kwargs: Additional blender-specific parameters:
            - frequency (float): Noise frequency for NoiseBlender (default: 0.05)
            - octaves (int): Noise octaves for NoiseBlender (default: 3)
            - levels (int): Pyramid levels for LaplacianBlender (default: 4)

    Returns:
        TileBlender: Instance of the requested blender class

    Raises:
        ValueError: If mode is not recognized

    Example:
        >>> blender = get_blender("linear", 32)
        >>> blender = get_blender("noise", 48, frequency=0.05, octaves=3)
        >>> blender = get_blender("laplacian", 64, levels=4)
    """
    if mode not in BLENDERS:
        available = list(BLENDERS.keys())
        raise ValueError(
            f"Unknown blend mode: '{mode}'. "
            f"Available modes: {available}"
        )

    return BLENDERS[mode](blend_width, **kwargs)


__all__ = [
    "TileBlender",
    "get_blender",
    "LinearBlender",
    "NoiseBlender",
    "LaplacianBlender",
]

__version__ = "2.0.0-dev"
