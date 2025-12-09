# Advanced Tile Blending for ComfyUI_SimpleTiles_Uprez

## Design Document v1.0

**Author:** Cedar  
**Date:** December 2024  
**Status:** Draft  
**Repository:** [ComfyUI_SimpleTiles_Uprez](https://github.com/cedarconnor/ComfyUI_SimpleTiles_Uprez)

---

## 1. Overview

### 1.1 Problem Statement

The current SimpleTiles implementation uses linear gradient blending in overlap regions. While functional, this approach produces predictable diagonal artifacts—subtle but visible grid patterns that the human eye easily detects, especially in:

- Textured regions (fabric, foliage, skin)
- Areas with consistent gradients (skies, backgrounds)
- High-frequency detail zones

### 1.2 Proposed Solution

Add a `blend_mode` dropdown to `DynamicTileSplit` and `DynamicTileMerge` nodes offering three blending strategies:

| Mode | Description | Use Case |
|------|-------------|----------|
| `linear` | Current implementation—simple gradient fade | Fast previews, low-detail images |
| `noise` | Soft Perlin/simplex noise boundaries | General-purpose, good quality/speed balance |
| `laplacian` | Multi-band Laplacian pyramid blending | Maximum quality, seamless results |

### 1.3 Goals

- **Backwards compatible**: Existing workflows continue to work unchanged
- **Minimal performance overhead**: `noise` mode adds <10% processing time
- **Seamless integration**: Single dropdown selection, no additional configuration required
- **Quality improvement**: Eliminate visible grid artifacts in upscaled outputs

---

## 2. Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      DynamicTileSplit                           │
│  ┌──────────────┐                                               │
│  │ blend_mode   │──► Stored in tile_calc object                 │
│  │  dropdown    │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   tile_calc     │
                    │  {              │
                    │    overlap_x,   │
                    │    overlap_y,   │
                    │    blend_mode,  │◄── NEW
                    │    ...          │
                    │  }              │
                    └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DynamicTileMerge                           │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│     ┌────────────┐   ┌────────────┐   ┌────────────┐           │
│     │  Linear    │   │   Noise    │   │ Laplacian  │           │
│     │  Blender   │   │  Blender   │   │  Blender   │           │
│     └────────────┘   └────────────┘   └────────────┘           │
│              │               │               │                  │
│              └───────────────┼───────────────┘                  │
│                              ▼                                  │
│                      Merged Output                              │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
ComfyUI_SimpleTiles_Uprez/
├── __init__.py
├── nodes.py
├── dynamic.py              # Modify: add blend_mode parameter
├── blending/               # NEW: blending module
│   ├── __init__.py
│   ├── base.py             # Abstract base class
│   ├── linear.py           # Current implementation extracted
│   ├── noise.py            # Perlin noise boundaries
│   ├── laplacian.py        # Multi-band pyramid blending
│   └── utils.py            # Shared utilities (pyramid ops, etc.)
└── standard.py
```

---

## 3. API Changes

### 3.1 DynamicTileSplit Node

```python
class DynamicTileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512}),
                # NEW PARAMETER
                "blend_mode": (["linear", "noise", "laplacian"], {"default": "linear"}),
            },
            "optional": {
                "offset": ("INT", {"default": 0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    RETURN_NAMES = ("tiles", "tile_calc")
```

### 3.2 tile_calc Object Extension

```python
tile_calc = {
    # Existing fields
    "overlap": overlap,
    "image_height": image_height,
    "image_width": image_width,
    "offset": offset,
    "tile_height": tile_height,
    "tile_width": tile_width,
    "rows": rows,
    "cols": cols,
    
    # New fields
    "blend_mode": blend_mode,           # "linear" | "noise" | "laplacian"
    "overlap_x": overlap_x,             # Calculated x-axis overlap
    "overlap_y": overlap_y,             # Calculated y-axis overlap
    "tile_positions": [                 # Tile placement metadata
        {
            "index": 0,
            "row": 0,
            "col": 0,
            "x1": 0, "y1": 0,           # Source crop coordinates
            "x2": 576, "y2": 576,
            "place_x": 0, "place_y": 0, # Destination placement
        },
        # ...
    ]
}
```

### 3.3 DynamicTileMerge Node

```python
class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
                "blend": ("INT", {"default": 32, "min": 0, "max": 256}),
            },
        }
    
    # blend_mode is read from tile_calc, not a separate input
    # This ensures split/merge consistency
```

---

## 4. Blending Implementations

### 4.1 Base Class

```python
# blending/base.py

from abc import ABC, abstractmethod
import torch

class TileBlender(ABC):
    """Abstract base class for tile blending strategies."""
    
    def __init__(self, blend_width: int):
        """
        Args:
            blend_width: Number of pixels over which blending occurs.
                        Must be <= overlap.
        """
        self.blend_width = blend_width
    
    @abstractmethod
    def create_mask(
        self, 
        height: int, 
        width: int, 
        direction: str,  # "horizontal" | "vertical"
        seed: int = 0
    ) -> torch.Tensor:
        """
        Create a blend mask for the overlap region.
        
        Returns:
            Tensor of shape (height, width) with values in [0, 1].
            0 = fully background, 1 = fully foreground tile.
        """
        pass
    
    @abstractmethod
    def blend_tiles(
        self,
        canvas: torch.Tensor,      # (B, H, W, C)
        tile: torch.Tensor,        # (B, tile_h, tile_w, C)
        position: dict,            # Tile placement info
        tile_calc: dict
    ) -> torch.Tensor:
        """
        Blend a single tile onto the canvas.
        
        Returns:
            Updated canvas with tile blended in.
        """
        pass
    
    def blend_overlap_region(
        self,
        background: torch.Tensor,  # (B, H, W, C)
        foreground: torch.Tensor,  # (B, H, W, C) - same size region
        mask: torch.Tensor         # (H, W)
    ) -> torch.Tensor:
        """Apply mask-based blending to overlapping regions."""
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        return background * (1 - mask) + foreground * mask
```

### 4.2 Linear Blender (Current Behavior)

```python
# blending/linear.py

import torch
from .base import TileBlender

class LinearBlender(TileBlender):
    """
    Simple linear gradient blending.
    Fast but produces visible grid artifacts.
    """
    
    def create_mask(
        self, 
        height: int, 
        width: int, 
        direction: str,
        seed: int = 0  # Unused for linear
    ) -> torch.Tensor:
        """Create a linear gradient mask."""
        
        if direction == "horizontal":
            # Gradient from 0 to 1 across width (for left-right seams)
            mask = torch.linspace(0, 1, width)
            mask = mask.unsqueeze(0).expand(height, -1)
        else:
            # Gradient from 0 to 1 across height (for top-bottom seams)
            mask = torch.linspace(0, 1, height)
            mask = mask.unsqueeze(1).expand(-1, width)
        
        return mask
    
    def blend_tiles(
        self,
        canvas: torch.Tensor,
        tile: torch.Tensor,
        position: dict,
        tile_calc: dict
    ) -> torch.Tensor:
        """Blend tile onto canvas using linear gradients."""
        
        blend_w = self.blend_width
        px, py = position["place_x"], position["place_y"]
        th, tw = tile.shape[1], tile.shape[2]
        
        # Determine which edges need blending
        has_left = position["col"] > 0
        has_top = position["row"] > 0
        
        # Start with direct placement
        result = canvas.clone()
        
        # Non-overlapping center region: direct copy
        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0
        
        result[:, py+cy1:py+th, px+cx1:px+tw, :] = tile[:, cy1:, cx1:, :]
        
        # Left edge blending
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py:py+th, px:px+blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py:py+th, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask)
        
        # Top edge blending
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py:py+blend_w, px:px+tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py:py+blend_w, px:px+tw, :] = self.blend_overlap_region(bg, fg, mask)
        
        # Corner: average the two blending operations
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            mask_corner = (mask_h + mask_v) / 2  # Simple average
            
            bg = canvas[:, py:py+blend_w, px:px+blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py:py+blend_w, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)
        
        return result
```

### 4.3 Noise Blender

```python
# blending/noise.py

import torch
import math
from .base import TileBlender

class NoiseBlender(TileBlender):
    """
    Perlin-style noise boundaries for organic blending.
    Breaks up regular grid artifacts with randomized seam positions.
    """
    
    def __init__(self, blend_width: int, frequency: float = 0.05, octaves: int = 3):
        """
        Args:
            blend_width: Blending region width in pixels.
            frequency: Base noise frequency. Lower = larger features.
            octaves: Number of noise layers. More = finer detail.
        """
        super().__init__(blend_width)
        self.frequency = frequency
        self.octaves = octaves
    
    def _smoothstep(self, t: torch.Tensor) -> torch.Tensor:
        """Smooth interpolation curve: 3t² - 2t³"""
        return t * t * (3 - 2 * t)
    
    def _perlin_1d(self, x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """
        Simple 1D Perlin-like noise.
        
        For each position, interpolate between random gradients
        at integer boundaries.
        """
        torch.manual_seed(seed)
        
        # Integer positions
        x0 = x.floor().long()
        x1 = x0 + 1
        
        # Fractional position
        t = x - x0.float()
        t = self._smoothstep(t)
        
        # Random gradients at integer positions (deterministic from seed)
        max_idx = x1.max().item() + 1
        gradients = torch.rand(max_idx + 1) * 2 - 1  # Range [-1, 1]
        
        # Gather gradients
        g0 = gradients[x0.clamp(0, max_idx)]
        g1 = gradients[x1.clamp(0, max_idx)]
        
        # Interpolate
        return g0 * (1 - t) + g1 * t
    
    def _fbm_1d(self, x: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """
        Fractional Brownian Motion: layered noise for natural appearance.
        """
        result = torch.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        
        for i in range(self.octaves):
            result += self._perlin_1d(x * frequency, seed + i * 1000) * amplitude
            amplitude *= 0.5
            frequency *= 2.0
        
        # Normalize to [0, 1]
        result = (result + 1) / 2
        return result.clamp(0, 1)
    
    def create_mask(
        self, 
        height: int, 
        width: int, 
        direction: str,
        seed: int = 0
    ) -> torch.Tensor:
        """
        Create a noise-modulated blend mask.
        
        The boundary position varies according to Perlin noise,
        with a soft gradient across the boundary.
        """
        
        softness = max(width // 4, 8)  # Gradient softness in pixels
        
        if direction == "horizontal":
            # Noise varies along Y-axis, boundary moves in X
            y = torch.arange(height).float() * self.frequency
            noise = self._fbm_1d(y, seed)
            
            # Noise determines where the boundary falls (0.3 to 0.7 of width)
            boundary = noise * 0.4 + 0.3  # Centered, doesn't touch edges
            boundary = boundary * width
            
            # Create soft transition around boundary
            x = torch.arange(width).float().unsqueeze(0).expand(height, -1)
            boundary = boundary.unsqueeze(1).expand(-1, width)
            
            # Signed distance from boundary, normalized by softness
            dist = (x - boundary) / softness
            mask = torch.sigmoid(dist)  # Smooth S-curve transition
            
        else:
            # Noise varies along X-axis, boundary moves in Y
            x = torch.arange(width).float() * self.frequency
            noise = self._fbm_1d(x, seed)
            
            boundary = noise * 0.4 + 0.3
            boundary = boundary * height
            
            y = torch.arange(height).float().unsqueeze(1).expand(-1, width)
            boundary = boundary.unsqueeze(0).expand(height, -1)
            
            dist = (y - boundary) / softness
            mask = torch.sigmoid(dist)
        
        return mask
    
    def blend_tiles(
        self,
        canvas: torch.Tensor,
        tile: torch.Tensor,
        position: dict,
        tile_calc: dict
    ) -> torch.Tensor:
        """Blend tile using noise-based boundaries."""
        
        blend_w = self.blend_width
        px, py = position["place_x"], position["place_y"]
        th, tw = tile.shape[1], tile.shape[2]
        idx = position["index"]
        
        has_left = position["col"] > 0
        has_top = position["row"] > 0
        
        result = canvas.clone()
        
        # Non-overlapping region
        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0
        result[:, py+cy1:py+th, px+cx1:px+tw, :] = tile[:, cy1:, cx1:, :]
        
        # Use tile index as seed for reproducible but varied noise
        base_seed = idx * 12345
        
        # Left edge
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal", seed=base_seed)
            bg = canvas[:, py:py+th, px:px+blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py:py+th, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask)
        
        # Top edge
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical", seed=base_seed + 1)
            bg = canvas[:, py:py+blend_w, px:px+tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py:py+blend_w, px:px+tw, :] = self.blend_overlap_region(bg, fg, mask)
        
        # Corner: multiplicative combination of both masks
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal", seed=base_seed)
            mask_v = self.create_mask(blend_w, blend_w, "vertical", seed=base_seed + 1)
            
            # Multiplicative gives proper corner weighting
            mask_corner = mask_h * mask_v
            
            bg = canvas[:, py:py+blend_w, px:px+blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py:py+blend_w, px:px+blend_w, :] = self.blend_overlap_region(bg, fg, mask_corner)
        
        return result
```

### 4.4 Laplacian Pyramid Blender

```python
# blending/laplacian.py

import torch
import torch.nn.functional as F
from .base import TileBlender

class LaplacianBlender(TileBlender):
    """
    Multi-band Laplacian pyramid blending.
    
    This is the gold standard for seamless blending, used in professional
    panorama stitching. It blends different frequency bands separately,
    which prevents both hard seams AND soft ghosting artifacts.
    
    Reference: Burt & Adelson, "A Multiresolution Spline With Application 
               to Image Mosaics", 1983
    """
    
    def __init__(self, blend_width: int, levels: int = 4):
        """
        Args:
            blend_width: Blending region width.
            levels: Number of pyramid levels. More = smoother blending
                   but slower. 4-6 is usually sufficient.
        """
        super().__init__(blend_width)
        self.levels = levels
    
    def _gaussian_kernel(self, size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """Create a 2D Gaussian kernel for pyramid operations."""
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return torch.outer(g, g)
    
    def _downsample(self, img: torch.Tensor) -> torch.Tensor:
        """
        Gaussian blur then subsample by 2.
        Input: (B, H, W, C) -> Output: (B, H//2, W//2, C)
        """
        # Convert to (B, C, H, W) for conv2d
        x = img.permute(0, 3, 1, 2)
        
        # Apply Gaussian blur
        kernel = self._gaussian_kernel(5, 1.0).to(img.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)
        kernel = kernel.expand(x.shape[1], -1, -1, -1)  # (C, 1, 5, 5)
        
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        
        # Subsample
        x = x[:, :, ::2, ::2]
        
        return x.permute(0, 2, 3, 1)  # Back to (B, H, W, C)
    
    def _upsample(self, img: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Upsample by 2 with Gaussian interpolation.
        """
        x = img.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Smooth after upsampling
        kernel = self._gaussian_kernel(5, 1.0).to(img.device) * 4  # Scale for energy preservation
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(x.shape[1], -1, -1, -1)
        
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=x.shape[1])
        
        return x.permute(0, 2, 3, 1)
    
    def _build_gaussian_pyramid(self, img: torch.Tensor) -> list:
        """Build Gaussian pyramid (progressively blurred/downsampled)."""
        pyramid = [img]
        current = img
        
        for _ in range(self.levels - 1):
            current = self._downsample(current)
            pyramid.append(current)
        
        return pyramid
    
    def _build_laplacian_pyramid(self, img: torch.Tensor) -> list:
        """
        Build Laplacian pyramid (band-pass filtered versions).
        
        Each level contains the detail lost when downsampling.
        The final level is the low-frequency residual.
        """
        gaussian = self._build_gaussian_pyramid(img)
        laplacian = []
        
        for i in range(len(gaussian) - 1):
            size = (gaussian[i].shape[1], gaussian[i].shape[2])
            upsampled = self._upsample(gaussian[i + 1], size)
            laplacian.append(gaussian[i] - upsampled)
        
        # Last level is the low-frequency residual
        laplacian.append(gaussian[-1])
        
        return laplacian
    
    def _collapse_laplacian_pyramid(self, pyramid: list) -> torch.Tensor:
        """Reconstruct image from Laplacian pyramid."""
        img = pyramid[-1]  # Start with low-frequency residual
        
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[2])
            img = self._upsample(img, size) + pyramid[i]
        
        return img
    
    def _build_mask_pyramid(self, mask: torch.Tensor) -> list:
        """Build Gaussian pyramid for the blend mask."""
        # Expand mask to (1, H, W, 1) format
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(-1)
        
        pyramid = [mask]
        current = mask
        
        for _ in range(self.levels - 1):
            current = self._downsample(current)
            pyramid.append(current)
        
        return pyramid
    
    def create_mask(
        self, 
        height: int, 
        width: int, 
        direction: str,
        seed: int = 0
    ) -> torch.Tensor:
        """
        Create a simple gradient mask for Laplacian blending.
        
        Note: With Laplacian blending, even a simple linear mask
        produces excellent results because each frequency band
        is blended separately.
        """
        if direction == "horizontal":
            mask = torch.linspace(0, 1, width)
            mask = mask.unsqueeze(0).expand(height, -1)
        else:
            mask = torch.linspace(0, 1, height)
            mask = mask.unsqueeze(1).expand(-1, width)
        
        return mask
    
    def blend_region_laplacian(
        self,
        background: torch.Tensor,  # (B, H, W, C)
        foreground: torch.Tensor,  # (B, H, W, C)
        mask: torch.Tensor         # (H, W)
    ) -> torch.Tensor:
        """
        Blend two regions using Laplacian pyramid.
        """
        # Build Laplacian pyramids for both images
        lap_bg = self._build_laplacian_pyramid(background)
        lap_fg = self._build_laplacian_pyramid(foreground)
        
        # Build Gaussian pyramid for mask
        mask_pyr = self._build_mask_pyramid(mask.to(background.device))
        
        # Blend each level
        blended_pyramid = []
        for l_bg, l_fg, m in zip(lap_bg, lap_fg, mask_pyr):
            # Expand mask if needed
            if m.shape[1:3] != l_bg.shape[1:3]:
                m = F.interpolate(
                    m.permute(0, 3, 1, 2), 
                    size=(l_bg.shape[1], l_bg.shape[2]),
                    mode='bilinear',
                    align_corners=False
                ).permute(0, 2, 3, 1)
            
            blended = l_bg * (1 - m) + l_fg * m
            blended_pyramid.append(blended)
        
        # Reconstruct
        return self._collapse_laplacian_pyramid(blended_pyramid)
    
    def blend_tiles(
        self,
        canvas: torch.Tensor,
        tile: torch.Tensor,
        position: dict,
        tile_calc: dict
    ) -> torch.Tensor:
        """Blend tile using Laplacian pyramid blending."""
        
        blend_w = self.blend_width
        px, py = position["place_x"], position["place_y"]
        th, tw = tile.shape[1], tile.shape[2]
        
        has_left = position["col"] > 0
        has_top = position["row"] > 0
        
        result = canvas.clone()
        
        # Non-overlapping region: direct copy
        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0
        result[:, py+cy1:py+th, px+cx1:px+tw, :] = tile[:, cy1:, cx1:, :]
        
        # Left edge with Laplacian blending
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py:py+th, px:px+blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py:py+th, px:px+blend_w, :] = self.blend_region_laplacian(bg, fg, mask)
        
        # Top edge
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py:py+blend_w, px:px+tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py:py+blend_w, px:px+tw, :] = self.blend_region_laplacian(bg, fg, mask)
        
        # Corner
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            mask_corner = (mask_h + mask_v) / 2
            
            bg = canvas[:, py:py+blend_w, px:px+blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py:py+blend_w, px:px+blend_w, :] = self.blend_region_laplacian(bg, fg, mask_corner)
        
        return result
```

### 4.5 Blender Factory

```python
# blending/__init__.py

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
    Factory function to create appropriate blender.
    
    Args:
        mode: One of "linear", "noise", "laplacian"
        blend_width: Blending region width in pixels
        **kwargs: Additional blender-specific parameters
    
    Returns:
        TileBlender instance
    """
    if mode not in BLENDERS:
        raise ValueError(f"Unknown blend mode: {mode}. Choose from {list(BLENDERS.keys())}")
    
    return BLENDERS[mode](blend_width, **kwargs)
```

---

## 5. Integration with Existing Code

### 5.1 Modified DynamicTileSplit

```python
# dynamic.py (modifications)

class DynamicTileSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tile_height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "tile_width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8}),
                "blend_mode": (["linear", "noise", "laplacian"], {"default": "noise"}),  # NEW
            },
            "optional": {
                "offset": ("INT", {"default": 0, "min": 0, "max": 256}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    RETURN_NAMES = ("tiles", "tile_calc")
    FUNCTION = "split"
    CATEGORY = "image/tiles"

    def split(self, image, tile_height, tile_width, overlap, blend_mode, offset=0):
        B, H, W, C = image.shape
        
        # Calculate grid dimensions (existing logic)
        # ...
        
        tiles = []
        tile_positions = []
        
        for row in range(rows):
            for col in range(cols):
                # Calculate tile coordinates with overlap
                y1 = max(0, row * step_y - (overlap if row > 0 else 0))
                x1 = max(0, col * step_x - (overlap if col > 0 else 0))
                y2 = min(H, y1 + tile_height + (overlap if row > 0 else 0))
                x2 = min(W, x1 + tile_width + (overlap if col > 0 else 0))
                
                tile = image[:, y1:y2, x1:x2, :]
                tiles.append(tile)
                
                tile_positions.append({
                    "index": row * cols + col,
                    "row": row,
                    "col": col,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "place_x": col * step_x,
                    "place_y": row * step_y,
                })
        
        tiles_tensor = torch.stack(tiles) if len(tiles) > 1 else tiles[0].unsqueeze(0)
        
        tile_calc = {
            "overlap": overlap,
            "overlap_x": overlap,  # Could differ for non-square aspect ratios
            "overlap_y": overlap,
            "image_height": H,
            "image_width": W,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "rows": rows,
            "cols": cols,
            "offset": offset,
            "blend_mode": blend_mode,  # NEW
            "tile_positions": tile_positions,  # NEW
        }
        
        return (tiles_tensor, tile_calc)
```

### 5.2 Modified DynamicTileMerge

```python
# dynamic.py (modifications)

from blending import get_blender

class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_calc": ("TILE_CALC",),
                "blend": ("INT", {"default": 32, "min": 0, "max": 256, "step": 8}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "merge"
    CATEGORY = "image/tiles"

    def merge(self, tiles, tile_calc, blend):
        H = tile_calc["image_height"]
        W = tile_calc["image_width"]
        blend_mode = tile_calc.get("blend_mode", "linear")  # Backwards compatible
        positions = tile_calc.get("tile_positions", None)
        
        B, C = tiles.shape[0], tiles.shape[-1]
        
        # Initialize canvas
        canvas = torch.zeros((1, H, W, C), dtype=tiles.dtype, device=tiles.device)
        
        # Get appropriate blender
        blender = get_blender(blend_mode, blend)
        
        # Process each tile
        for i in range(tiles.shape[0]):
            tile = tiles[i:i+1]  # Keep batch dimension
            
            if positions:
                position = positions[i]
            else:
                # Backwards compatibility: reconstruct position
                row = i // tile_calc["cols"]
                col = i % tile_calc["cols"]
                position = {
                    "index": i,
                    "row": row,
                    "col": col,
                    "place_x": col * (tile_calc["tile_width"] - tile_calc["overlap"]),
                    "place_y": row * (tile_calc["tile_height"] - tile_calc["overlap"]),
                }
            
            canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
        
        return (canvas,)
```

---

## 6. Performance Considerations

### 6.1 Benchmarks (Estimated)

| Mode | Relative Speed | Memory Overhead | Quality |
|------|---------------|-----------------|---------|
| `linear` | 1.0x (baseline) | None | Good |
| `noise` | 1.05-1.1x | Minimal | Very Good |
| `laplacian` | 1.5-2.0x | ~4x blend region | Excellent |

### 6.2 Optimization Opportunities

**Noise Blender:**
- Pre-compute noise LUTs for common dimensions
- Use GPU-accelerated noise generation (if available)

**Laplacian Blender:**
- Cache Gaussian kernels
- Use separable convolutions (2x speedup)
- Limit pyramid levels based on blend width (no need for 6 levels on 32px overlap)
- Process all tiles in batch where possible

```python
# Optimized separable Gaussian blur
def _gaussian_blur_separable(self, img, kernel_1d):
    """2x faster than 2D convolution."""
    x = img.permute(0, 3, 1, 2)
    
    # Horizontal pass
    k_h = kernel_1d.view(1, 1, 1, -1)
    x = F.conv2d(x, k_h.expand(x.shape[1], -1, -1, -1), 
                 padding=(0, len(kernel_1d)//2), groups=x.shape[1])
    
    # Vertical pass  
    k_v = kernel_1d.view(1, 1, -1, 1)
    x = F.conv2d(x, k_v.expand(x.shape[1], -1, -1, -1),
                 padding=(len(kernel_1d)//2, 0), groups=x.shape[1])
    
    return x.permute(0, 2, 3, 1)
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# tests/test_blending.py

import pytest
import torch
from blending import get_blender, LinearBlender, NoiseBlender, LaplacianBlender

class TestBlenderFactory:
    def test_get_linear(self):
        blender = get_blender("linear", 32)
        assert isinstance(blender, LinearBlender)
    
    def test_get_noise(self):
        blender = get_blender("noise", 32)
        assert isinstance(blender, NoiseBlender)
    
    def test_get_laplacian(self):
        blender = get_blender("laplacian", 32)
        assert isinstance(blender, LaplacianBlender)
    
    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            get_blender("invalid", 32)


class TestMaskGeneration:
    @pytest.mark.parametrize("blender_class", [LinearBlender, NoiseBlender, LaplacianBlender])
    def test_mask_shape(self, blender_class):
        blender = blender_class(32)
        mask = blender.create_mask(128, 64, "horizontal")
        assert mask.shape == (128, 64)
    
    @pytest.mark.parametrize("blender_class", [LinearBlender, NoiseBlender, LaplacianBlender])
    def test_mask_range(self, blender_class):
        blender = blender_class(32)
        mask = blender.create_mask(128, 64, "horizontal")
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_noise_reproducibility(self):
        blender = NoiseBlender(32)
        mask1 = blender.create_mask(128, 64, "horizontal", seed=42)
        mask2 = blender.create_mask(128, 64, "horizontal", seed=42)
        assert torch.allclose(mask1, mask2)
    
    def test_noise_variation(self):
        blender = NoiseBlender(32)
        mask1 = blender.create_mask(128, 64, "horizontal", seed=1)
        mask2 = blender.create_mask(128, 64, "horizontal", seed=2)
        assert not torch.allclose(mask1, mask2)


class TestBlending:
    @pytest.fixture
    def sample_tiles(self):
        # Create simple gradient tiles for testing
        tile1 = torch.zeros(1, 64, 64, 3)
        tile1[:, :, :, 0] = 1.0  # Red
        
        tile2 = torch.zeros(1, 64, 64, 3)
        tile2[:, :, :, 2] = 1.0  # Blue
        
        return tile1, tile2
    
    @pytest.mark.parametrize("mode", ["linear", "noise", "laplacian"])
    def test_blend_preserves_shape(self, mode, sample_tiles):
        tile1, tile2 = sample_tiles
        blender = get_blender(mode, 16)
        
        mask = blender.create_mask(64, 16, "horizontal")
        result = blender.blend_overlap_region(tile1[:, :, :16, :], tile2[:, :, :16, :], mask)
        
        assert result.shape == (1, 64, 16, 3)
    
    @pytest.mark.parametrize("mode", ["linear", "noise", "laplacian"])
    def test_blend_interpolates(self, mode, sample_tiles):
        tile1, tile2 = sample_tiles
        blender = get_blender(mode, 16)
        
        mask = blender.create_mask(64, 16, "horizontal")
        result = blender.blend_overlap_region(tile1[:, :, :16, :], tile2[:, :, :16, :], mask)
        
        # Result should have mixed values (not pure red or blue)
        assert result[:, :, 8, :].mean() > 0.1  # Some mixing in middle
        assert result[:, :, 8, :].mean() < 0.9


class TestLaplacianPyramid:
    def test_pyramid_levels(self):
        blender = LaplacianBlender(32, levels=4)
        img = torch.rand(1, 128, 128, 3)
        pyramid = blender._build_laplacian_pyramid(img)
        
        assert len(pyramid) == 4
        assert pyramid[0].shape == (1, 128, 128, 3)
        assert pyramid[1].shape == (1, 64, 64, 3)
        assert pyramid[2].shape == (1, 32, 32, 3)
        assert pyramid[3].shape == (1, 16, 16, 3)
    
    def test_pyramid_reconstruction(self):
        blender = LaplacianBlender(32, levels=4)
        img = torch.rand(1, 128, 128, 3)
        pyramid = blender._build_laplacian_pyramid(img)
        reconstructed = blender._collapse_laplacian_pyramid(pyramid)
        
        # Should reconstruct approximately (some loss due to filtering)
        assert torch.allclose(img, reconstructed, atol=0.01)
```

### 7.2 Visual Integration Tests

```python
# tests/test_visual.py

import torch
from PIL import Image
import numpy as np

def test_full_pipeline_visual():
    """Generate comparison images for manual inspection."""
    
    # Create a test image with challenging features
    test_img = create_test_pattern(512, 512)  # Gradients + texture
    
    for mode in ["linear", "noise", "laplacian"]:
        # Split
        splitter = DynamicTileSplit()
        tiles, tile_calc = splitter.split(test_img, 256, 256, 64, mode)
        
        # Merge
        merger = DynamicTileMerge()
        result, = merger.merge(tiles, tile_calc, 48)
        
        # Save for inspection
        save_image(result, f"test_output_{mode}.png")
        
        # Generate difference image (amplified)
        diff = (result - test_img).abs() * 10
        save_image(diff, f"test_diff_{mode}.png")

def create_test_pattern(h, w):
    """Create image with gradients and textures to reveal seams."""
    img = torch.zeros(1, h, w, 3)
    
    # Diagonal gradient (reveals linear seams)
    for y in range(h):
        for x in range(w):
            img[0, y, x, 0] = (x + y) / (h + w)
    
    # High-frequency noise (reveals smoothing artifacts)
    img[:, :, :, 1] = torch.rand(1, h, w) * 0.3
    
    # Circular pattern (reveals discontinuities)
    cy, cx = h // 2, w // 2
    for y in range(h):
        for x in range(w):
            r = ((x - cx)**2 + (y - cy)**2) ** 0.5
            img[0, y, x, 2] = (np.sin(r * 0.1) + 1) / 2 * 0.5
    
    return img
```

---

## 8. Migration & Backwards Compatibility

### 8.1 Existing Workflow Compatibility

Existing workflows will continue to work without modification:

1. If `blend_mode` is not specified in `DynamicTileSplit`, it defaults to `"linear"`
2. If `tile_calc` doesn't contain `blend_mode`, `DynamicTileMerge` defaults to `"linear"`
3. If `tile_positions` is missing, positions are reconstructed from grid dimensions

### 8.2 Version Detection

```python
def get_tile_calc_version(tile_calc: dict) -> int:
    """Detect tile_calc format version."""
    if "blend_mode" in tile_calc and "tile_positions" in tile_calc:
        return 2  # New format
    return 1  # Legacy format
```

---

## 9. Future Enhancements

### 9.1 Potential Additions

- **Adaptive blending**: Automatically choose blend mode based on image content
- **Seam carving**: Find optimal seam path through low-energy regions
- **Content-aware blending**: Use semantic segmentation to avoid cutting objects
- **GPU acceleration**: CUDA kernels for Laplacian pyramid operations

### 9.2 User-Configurable Parameters

Could expose as "advanced" options in future:

```python
"optional": {
    "noise_frequency": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2}),
    "noise_octaves": ("INT", {"default": 3, "min": 1, "max": 6}),
    "pyramid_levels": ("INT", {"default": 4, "min": 2, "max": 8}),
}
```

---

## 10. Implementation Checklist

- [ ] Create `blending/` module structure
- [ ] Implement `LinearBlender` (extract from existing code)
- [ ] Implement `NoiseBlender`
- [ ] Implement `LaplacianBlender`
- [ ] Add `blend_mode` parameter to `DynamicTileSplit`
- [ ] Update `tile_calc` object structure
- [ ] Modify `DynamicTileMerge` to use blender factory
- [ ] Add backwards compatibility handling
- [ ] Write unit tests
- [ ] Write visual integration tests
- [ ] Update README.md documentation
- [ ] Test with real upscaling workflows
- [ ] Performance profiling and optimization

---

## Appendix A: References

1. Burt, P. J., & Adelson, E. H. (1983). "A Multiresolution Spline With Application to Image Mosaics." ACM Transactions on Graphics.

2. Pérez, P., Gangnet, M., & Blake, A. (2003). "Poisson Image Editing." ACM SIGGRAPH.

3. Perlin, K. (1985). "An Image Synthesizer." ACM SIGGRAPH.

---

## Appendix B: Visual Comparison

```
┌─────────────────┬─────────────────┬─────────────────┐
│     Linear      │      Noise      │    Laplacian    │
├─────────────────┼─────────────────┼─────────────────┤
│                 │                 │                 │
│  ═══════════    │  ∼∼∼∼∼∼∼∼∼∼∼    │  ≈≈≈≈≈≈≈≈≈≈≈    │
│  Visible grid   │  Organic seams  │  Invisible      │
│  artifacts      │  harder to see  │  transitions    │
│                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
```
