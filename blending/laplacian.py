"""
Laplacian Pyramid Blending

This module implements the LaplacianBlender class, which uses multi-band
pyramid blending for seamless tile merging. This is the highest quality
blending method, producing professional-grade results with invisible seams.

Based on: Burt & Adelson, "A Multiresolution Spline With Application to
Image Mosaics", ACM Transactions on Graphics, 1983.

Characteristics:
- Highest quality, seamless results
- Eliminates both hard seams AND soft ghosting
- Blends different frequency bands separately
- Moderate computational cost (~1.5-2× linear)
- Best for final renders and maximum quality

The multi-band approach prevents artifacts by blending high frequencies
sharply and low frequencies gradually, matching how the human visual system
processes images.
"""

import torch
import torch.nn.functional as F
from .base import TileBlender


class LaplacianBlender(TileBlender):
    """
    Multi-band Laplacian pyramid blending for seamless tile merging.

    This is the gold standard for image blending, used in professional
    panorama stitching and compositing. It works by:

    1. Decomposing images into frequency bands (Laplacian pyramid)
    2. Blending each frequency band separately using the mask
    3. Reconstructing the final result from all bands

    This approach eliminates visible seams because:
    - High frequencies (details) are blended with sharp transitions
    - Low frequencies (gradients) are blended with smooth transitions
    - The result matches natural image statistics

    Args:
        blend_width: Number of pixels over which blending occurs
        levels: Number of pyramid levels (default: 4)
               More levels = smoother blending but slower
               Typical range: 3-6

    Example:
        >>> blender = LaplacianBlender(blend_width=64, levels=4)
        >>> mask = blender.create_mask(256, 64, "horizontal")
        >>> canvas = blender.blend_tiles(canvas, tile, position, tile_calc)
    """

    def __init__(self, blend_width: int, levels: int = 4):
        """
        Initialize Laplacian pyramid blender.

        Args:
            blend_width: Blending region width in pixels
            levels: Number of pyramid levels (default: 4)
                   Higher values create smoother blending
                   Practical range: 2-8

        Raises:
            ValueError: If blend_width < 0 or levels < 2
        """
        super().__init__(blend_width)

        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")

        self.levels = levels
        self._gaussian_kernel_cache = {}  # Cache kernels for efficiency

    def _gaussian_kernel(self, size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """
        Create a 2D Gaussian kernel for pyramid operations.

        Uses the separable property: 2D Gaussian = outer product of 1D Gaussians.
        Kernels are cached for performance.

        Formula: G(x) = exp(-x²/2σ²) / √(2πσ²)

        Args:
            size: Kernel size (should be odd, default: 5)
            sigma: Standard deviation (default: 1.0)

        Returns:
            Normalized 2D Gaussian kernel of shape (size, size)

        Example:
            >>> kernel = blender._gaussian_kernel(5, 1.0)
            >>> kernel.shape, kernel.sum()
            (torch.Size([5, 5]), tensor(1.0))
        """
        cache_key = (size, sigma)
        if cache_key in self._gaussian_kernel_cache:
            return self._gaussian_kernel_cache[cache_key]

        # Create 1D Gaussian
        coords = torch.arange(size).float() - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()  # Normalize

        # Create 2D kernel via outer product
        kernel = torch.outer(g, g)

        # Cache for future use
        self._gaussian_kernel_cache[cache_key] = kernel

        return kernel

    def _downsample(self, img: torch.Tensor) -> torch.Tensor:
        """
        Gaussian blur then subsample by 2.

        This is the REDUCE operation in pyramid literature. It prevents
        aliasing by low-pass filtering before decimation.

        Args:
            img: Input tensor of shape (B, H, W, C)

        Returns:
            Downsampled tensor of shape (B, H//2, W//2, C)

        Implementation:
            1. Convert to BCHW format for conv2d
            2. Apply Gaussian blur (anti-aliasing filter)
            3. Subsample by taking every other pixel
            4. Convert back to BHWC format

        Example:
            >>> img = torch.rand(1, 256, 256, 3)
            >>> down = blender._downsample(img)
            >>> down.shape
            torch.Size([1, 128, 128, 3])
        """
        # Convert BHWC → BCHW for PyTorch conv2d
        x = img.permute(0, 3, 1, 2)

        # Get Gaussian kernel
        kernel = self._gaussian_kernel(5, 1.0).to(img.device)

        # Expand kernel to (C, 1, 5, 5) for depthwise convolution
        C = x.shape[1]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)
        kernel = kernel.expand(C, 1, -1, -1)  # (C, 1, 5, 5)

        # Apply Gaussian blur with reflection padding
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=C)

        # Subsample by 2 (take every other pixel)
        x = x[:, :, ::2, ::2]

        # Convert back to BHWC
        return x.permute(0, 2, 3, 1)

    def _upsample(self, img: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """
        Upsample by 2 with Gaussian interpolation.

        This is the EXPAND operation in pyramid literature. It creates
        a smooth upsampled version without introducing high-frequency
        artifacts.

        Args:
            img: Input tensor of shape (B, H, W, C)
            target_size: Target (height, width) after upsampling

        Returns:
            Upsampled tensor of shape (B, target_size[0], target_size[1], C)

        Implementation:
            1. Bilinear interpolation to target size
            2. Gaussian blur to smooth
            3. Scale by 4 to preserve energy

        Example:
            >>> img = torch.rand(1, 64, 64, 3)
            >>> up = blender._upsample(img, (128, 128))
            >>> up.shape
            torch.Size([1, 128, 128, 3])
        """
        # Convert BHWC → BCHW
        x = img.permute(0, 3, 1, 2)

        # Bilinear interpolation
        x = F.interpolate(
            x,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )

        # Get Gaussian kernel (scaled for energy preservation)
        kernel = self._gaussian_kernel(5, 1.0).to(img.device) * 4
        C = x.shape[1]
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

        # Apply Gaussian smoothing
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=C)

        # Convert back to BHWC
        return x.permute(0, 2, 3, 1)

    def _build_gaussian_pyramid(self, img: torch.Tensor) -> list:
        """
        Build Gaussian pyramid (progressively blurred/downsampled).

        Each level is a smoothed and downsampled version of the previous.
        This creates a multi-scale representation from fine to coarse.

        Args:
            img: Input image of shape (B, H, W, C)

        Returns:
            List of tensors, each half the size of the previous:
            [original, half, quarter, eighth, ...]

        Example:
            >>> img = torch.rand(1, 256, 256, 3)
            >>> pyramid = blender._build_gaussian_pyramid(img)
            >>> [p.shape for p in pyramid]
            [torch.Size([1, 256, 256, 3]),
             torch.Size([1, 128, 128, 3]),
             torch.Size([1, 64, 64, 3]),
             torch.Size([1, 32, 32, 3])]
        """
        pyramid = [img]
        current = img

        for _ in range(self.levels - 1):
            current = self._downsample(current)
            pyramid.append(current)

        return pyramid

    def _build_laplacian_pyramid(self, img: torch.Tensor) -> list:
        """
        Build Laplacian pyramid (band-pass filtered versions).

        Each level contains the detail lost when downsampling. This
        represents the image as a set of frequency bands. The last
        level is the low-frequency residual.

        Formula: L[i] = G[i] - upsample(G[i+1])

        Args:
            img: Input image of shape (B, H, W, C)

        Returns:
            List of tensors representing frequency bands:
            [high_freq, mid_freq, ..., low_freq_residual]

        The Laplacian pyramid is invertible: we can perfectly reconstruct
        the original by collapsing the pyramid.

        Example:
            >>> img = torch.rand(1, 128, 128, 3)
            >>> lap_pyr = blender._build_laplacian_pyramid(img)
            >>> # Each level captures detail at different scales
            >>> reconstructed = blender._collapse_laplacian_pyramid(lap_pyr)
            >>> torch.allclose(img, reconstructed, atol=0.01)
            True
        """
        # First build Gaussian pyramid
        gaussian = self._build_gaussian_pyramid(img)
        laplacian = []

        # Compute difference between each level and upsampled next level
        for i in range(len(gaussian) - 1):
            size = (gaussian[i].shape[1], gaussian[i].shape[2])
            upsampled = self._upsample(gaussian[i + 1], size)

            # Laplacian = current - upsampled_next
            # This captures the detail lost in downsampling
            laplacian.append(gaussian[i] - upsampled)

        # Last level is the low-frequency residual
        laplacian.append(gaussian[-1])

        return laplacian

    def _collapse_laplacian_pyramid(self, pyramid: list) -> torch.Tensor:
        """
        Reconstruct image from Laplacian pyramid.

        This inverts the pyramid construction by progressively upsampling
        and adding detail layers from coarse to fine.

        Args:
            pyramid: List of Laplacian levels from build_laplacian_pyramid

        Returns:
            Reconstructed image of shape (B, H, W, C)

        Algorithm:
            Start with lowest frequency
            For each level from coarse to fine:
                Upsample current result
                Add detail layer
            Return final result

        Example:
            >>> img = torch.rand(1, 128, 128, 3)
            >>> pyr = blender._build_laplacian_pyramid(img)
            >>> reconstructed = blender._collapse_laplacian_pyramid(pyr)
            >>> torch.allclose(img, reconstructed, atol=0.01)
            True
        """
        # Start with low-frequency residual (last level)
        img = pyramid[-1]

        # Progressively upsample and add detail
        for i in range(len(pyramid) - 2, -1, -1):
            size = (pyramid[i].shape[1], pyramid[i].shape[2])
            img = self._upsample(img, size) + pyramid[i]

        return img

    def _build_mask_pyramid(self, mask: torch.Tensor) -> list:
        """
        Build Gaussian pyramid for the blend mask.

        The mask pyramid is used to blend corresponding levels of the
        image pyramids. Lower frequency bands use more blurred masks.

        Args:
            mask: Blend mask of shape (H, W)

        Returns:
            List of progressively downsampled masks

        Example:
            >>> mask = torch.linspace(0, 1, 64).unsqueeze(0).expand(64, 64)
            >>> mask_pyr = blender._build_mask_pyramid(mask)
            >>> [m.shape for m in mask_pyr]
            [torch.Size([1, 64, 64, 1]),
             torch.Size([1, 32, 32, 1]),
             torch.Size([1, 16, 16, 1]),
             torch.Size([1, 8, 8, 1])]
        """
        # Convert mask to (1, H, W, 1) format
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
        seed: int = 0  # Unused for Laplacian
    ) -> torch.Tensor:
        """
        Create a simple gradient mask for Laplacian blending.

        With Laplacian blending, even a simple linear mask produces
        excellent results because each frequency band is blended
        separately. The multi-band approach handles the complexity.

        Args:
            height: Mask height in pixels
            width: Mask width in pixels
            direction: Blending direction ("horizontal" or "vertical")
            seed: Ignored (kept for API compatibility)

        Returns:
            Linear gradient mask of shape (height, width)

        Note:
            Unlike noise blending, Laplacian blending doesn't need
            complex masks. The pyramid decomposition does the heavy
            lifting to create seamless results.

        Example:
            >>> mask = blender.create_mask(128, 64, "horizontal")
            >>> mask.shape
            torch.Size([128, 64])
        """
        if direction == "horizontal":
            mask = torch.linspace(0, 1, width)
            mask = mask.unsqueeze(0).expand(height, -1)
        elif direction == "vertical":
            mask = torch.linspace(0, 1, height)
            mask = mask.unsqueeze(1).expand(-1, width)
        else:
            raise ValueError(
                f"Invalid direction: {direction}. Must be 'horizontal' or 'vertical'"
            )

        return mask

    def blend_region_laplacian(
        self,
        background: torch.Tensor,
        foreground: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Blend two regions using Laplacian pyramid.

        This is the core of multi-band blending. Each frequency band
        is blended separately, then all bands are combined to produce
        the final seamless result.

        Args:
            background: Background tile (B, H, W, C)
            foreground: Foreground tile (B, H, W, C)
            mask: Blend mask (H, W) with values in [0, 1]

        Returns:
            Blended result (B, H, W, C)

        Algorithm:
            1. Build Laplacian pyramids for both images
            2. Build Gaussian pyramid for mask
            3. Blend each frequency band using corresponding mask level
            4. Reconstruct final image from blended pyramid

        Example:
            >>> bg = torch.ones(1, 128, 128, 3) * 0.2
            >>> fg = torch.ones(1, 128, 128, 3) * 0.8
            >>> mask = torch.linspace(0, 1, 128).unsqueeze(0).expand(128, 128)
            >>> result = blender.blend_region_laplacian(bg, fg, mask)
            >>> result.shape
            torch.Size([1, 128, 128, 3])
        """
        # Build Laplacian pyramids for both images
        lap_bg = self._build_laplacian_pyramid(background)
        lap_fg = self._build_laplacian_pyramid(foreground)

        # Build Gaussian pyramid for mask
        mask_pyr = self._build_mask_pyramid(mask.to(background.device))

        # Blend each pyramid level
        blended_pyramid = []

        for l_bg, l_fg, m in zip(lap_bg, lap_fg, mask_pyr):
            # Ensure mask matches level size
            if m.shape[1:3] != l_bg.shape[1:3]:
                # Resize mask if dimensions don't match
                m_temp = m.permute(0, 3, 1, 2)  # BHWC → BCHW
                m_temp = F.interpolate(
                    m_temp,
                    size=(l_bg.shape[1], l_bg.shape[2]),
                    mode='bilinear',
                    align_corners=False
                )
                m = m_temp.permute(0, 2, 3, 1)  # BCHW → BHWC

            # Blend this frequency band
            # Formula: blended = background * (1 - mask) + foreground * mask
            blended = l_bg * (1 - m) + l_fg * m
            blended_pyramid.append(blended)

        # Reconstruct final image from blended pyramid
        result = self._collapse_laplacian_pyramid(blended_pyramid)

        # Clamp to valid range to prevent pyramid reconstruction artifacts
        # Laplacian pyramids can produce small overshoots on hard edges
        return result.clamp(0.0, 1.0)

    def blend_tiles(
        self,
        canvas: torch.Tensor,
        tile: torch.Tensor,
        position: dict,
        tile_calc: dict
    ) -> torch.Tensor:
        """
        Blend a tile onto canvas using Laplacian pyramid blending.

        Uses the highest quality multi-band blending for seamless results.
        This method is slower than linear or noise but produces the best
        visual quality.

        Args:
            canvas: Output canvas (B, H, W, C)
            tile: Tile to blend (B, tile_h, tile_w, C)
            position: Tile placement metadata
            tile_calc: Global tiling metadata

        Returns:
            Updated canvas with tile seamlessly blended

        Implementation:
            - Non-overlapping center: Direct copy
            - Left edge: Laplacian pyramid blend
            - Top edge: Laplacian pyramid blend
            - Corner: Averaged mask + Laplacian blend

        Example:
            >>> blender = LaplacianBlender(64, levels=4)
            >>> canvas = torch.zeros(1, 1024, 1024, 3)
            >>> tile = torch.rand(1, 256, 256, 3)
            >>> position = {"row": 1, "col": 1, "place_x": 192, "place_y": 192}
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
            px = position.get("x1", 0)
            py = position.get("y1", 0)

        # Determine which edges need blending
        row = position.get("row", 0)
        col = position.get("col", 0)
        has_left = col > 0
        has_top = row > 0

        # Clamp blend width
        blend_w = min(blend_w, th, tw)

        # Start with canvas copy
        result = canvas.clone()

        # Calculate non-overlapping center region
        cx1 = blend_w if has_left else 0
        cy1 = blend_w if has_top else 0

        # Copy non-overlapping center directly
        if cy1 < th and cx1 < tw:
            result[:, py+cy1:py+th, px+cx1:px+tw, :] = tile[:, cy1:, cx1:, :]

        # Blend left edge with Laplacian pyramid
        if has_left and blend_w > 0:
            mask = self.create_mask(th, blend_w, "horizontal")
            bg = canvas[:, py:py+th, px:px+blend_w, :]
            fg = tile[:, :, :blend_w, :]
            result[:, py:py+th, px:px+blend_w, :] = self.blend_region_laplacian(bg, fg, mask)

        # Blend top edge with Laplacian pyramid
        if has_top and blend_w > 0:
            mask = self.create_mask(blend_w, tw, "vertical")
            bg = canvas[:, py:py+blend_w, px:px+tw, :]
            fg = tile[:, :blend_w, :, :]
            result[:, py:py+blend_w, px:px+tw, :] = self.blend_region_laplacian(bg, fg, mask)

        # Blend corner with averaged mask
        if has_left and has_top and blend_w > 0:
            mask_h = self.create_mask(blend_w, blend_w, "horizontal")
            mask_v = self.create_mask(blend_w, blend_w, "vertical")
            mask_corner = (mask_h + mask_v) / 2

            bg = canvas[:, py:py+blend_w, px:px+blend_w, :]
            fg = tile[:, :blend_w, :blend_w, :]
            result[:, py:py+blend_w, px:px+blend_w, :] = self.blend_region_laplacian(bg, fg, mask_corner)

        return result

    def __repr__(self) -> str:
        """String representation of LaplacianBlender."""
        return f"LaplacianBlender(blend_width={self.blend_width}, levels={self.levels})"
