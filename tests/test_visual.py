"""
Visual Integration Tests for Tile Blending

Generates test images to manually inspect blending quality across different modes.
Creates comparison images showing artifacts and differences.

Run with: python tests/test_visual.py
"""

import torch
import math
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Imports will be enabled as components are implemented
# from blending import get_blender
# from dynamic import DynamicTileSplit, DynamicTileMerge


def create_test_pattern(height: int, width: int) -> torch.Tensor:
    """
    Create a challenging test pattern that reveals blending artifacts.

    The pattern includes:
    - Diagonal gradient (reveals linear seam artifacts)
    - High-frequency noise (reveals smoothing artifacts)
    - Circular pattern (reveals discontinuities)

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Tensor of shape (1, height, width, 3) with RGB test pattern
    """
    img = torch.zeros(1, height, width, 3)

    # Channel 0 (Red): Diagonal gradient
    # This creates a smooth gradient from top-left to bottom-right
    # Linear blending will show visible diagonal seams
    for y in range(height):
        for x in range(width):
            img[0, y, x, 0] = (x + y) / (height + width)

    # Channel 1 (Green): High-frequency noise
    # Random noise reveals if blending smooths too much
    img[:, :, :, 1] = torch.rand(1, height, width) * 0.3

    # Channel 2 (Blue): Circular pattern
    # Concentric circles reveal discontinuities at seams
    cy, cx = height // 2, width // 2
    for y in range(height):
        for x in range(width):
            r = math.sqrt((x - cx)**2 + (y - cy)**2)
            img[0, y, x, 2] = (math.sin(r * 0.1) + 1) / 2 * 0.5

    return img


def create_gradient_pattern(height: int, width: int) -> torch.Tensor:
    """
    Create a simple gradient pattern (sky simulation).

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Tensor of shape (1, height, width, 3) with gradient
    """
    img = torch.zeros(1, height, width, 3)

    for y in range(height):
        value = y / height
        img[0, y, :, 0] = 0.5 + value * 0.3  # Red
        img[0, y, :, 1] = 0.7 + value * 0.2  # Green
        img[0, y, :, 2] = 0.9 - value * 0.3  # Blue (sky gradient)

    return img


def create_texture_pattern(height: int, width: int) -> torch.Tensor:
    """
    Create a textured pattern (fabric/foliage simulation).

    Args:
        height: Image height in pixels
        width: Image width in pixels

    Returns:
        Tensor of shape (1, height, width, 3) with texture
    """
    img = torch.zeros(1, height, width, 3)

    # Base color
    img[:, :, :, :] = 0.4

    # Add noise texture
    noise = torch.rand(1, height, width, 3) * 0.3
    img += noise

    # Add some structure
    for y in range(0, height, 8):
        img[0, y:y+2, :, :] *= 1.2

    return img.clamp(0, 1)


def save_tensor_as_image(tensor: torch.Tensor, path: str):
    """
    Save a tensor as an image file.

    Args:
        tensor: Tensor of shape (1, H, W, C) with values in [0, 1]
        path: Output file path
    """
    try:
        from PIL import Image
        import numpy as np

        # Convert to numpy and scale to [0, 255]
        arr = (tensor[0].cpu().numpy() * 255).astype('uint8')

        # Create PIL image
        if arr.shape[-1] == 1:
            img = Image.fromarray(arr[:, :, 0], mode='L')
        elif arr.shape[-1] == 3:
            img = Image.fromarray(arr, mode='RGB')
        elif arr.shape[-1] == 4:
            img = Image.fromarray(arr, mode='RGBA')
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape[-1]}")

        # Save
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img.save(path)
        print(f"Saved: {path}")

    except ImportError:
        print("WARNING: PIL not available, cannot save images")
        print(f"Would have saved: {path}")


def test_full_pipeline_visual():
    """
    Test complete tile split/merge pipeline with all blend modes.

    Generates comparison images for manual inspection.
    """
    print("\n=== Visual Integration Test ===\n")
    print("NOTE: This test requires LinearBlender to be implemented")
    print("Skipping actual test execution until implementation is complete\n")

    # TODO: Uncomment when blenders are implemented
    """
    output_dir = Path(__file__).parent / "visual_output"
    output_dir.mkdir(exist_ok=True)

    # Test configurations
    image_size = 512
    tile_size = 256
    overlap = 64
    blend_width = 48

    test_patterns = [
        ("complex", create_test_pattern(image_size, image_size)),
        ("gradient", create_gradient_pattern(image_size, image_size)),
        ("texture", create_texture_pattern(image_size, image_size)),
    ]

    for pattern_name, test_img in test_patterns:
        print(f"\nTesting pattern: {pattern_name}")

        # Save original
        save_tensor_as_image(
            test_img,
            str(output_dir / f"{pattern_name}_original.png")
        )

        for mode in ["linear", "noise", "laplacian"]:
            print(f"  Processing with {mode} blending...")

            # Split
            splitter = DynamicTileSplit()
            tiles, tile_calc = splitter.split(
                test_img, tile_size, tile_size, overlap, mode
            )

            # Merge (simulating processing by just using tiles as-is)
            merger = DynamicTileMerge()
            result, = merger.merge(tiles, tile_calc, blend_width)

            # Save result
            save_tensor_as_image(
                result,
                str(output_dir / f"{pattern_name}_{mode}.png")
            )

            # Generate difference image (amplified)
            diff = (result - test_img).abs() * 10
            save_tensor_as_image(
                diff,
                str(output_dir / f"{pattern_name}_{mode}_diff.png")
            )

    print(f"\nVisual test outputs saved to: {output_dir}")
    print("Please inspect images manually to evaluate blending quality")
    """


def test_seam_visibility():
    """
    Test specifically for seam visibility with different blend modes.

    Creates a zoomed view of tile boundaries.
    """
    print("\n=== Seam Visibility Test ===")
    print("Skipped until implementation is complete\n")

    # TODO: Implement when blenders are ready


def test_performance_benchmark():
    """
    Benchmark processing time for different blend modes.
    """
    print("\n=== Performance Benchmark ===")
    print("Skipped until implementation is complete\n")

    # TODO: Implement when blenders are ready
    """
    import time

    image_size = 2048
    tile_size = 512
    overlap = 64
    blend_width = 48
    iterations = 10

    test_img = create_test_pattern(image_size, image_size)

    results = {}

    for mode in ["linear", "noise", "laplacian"]:
        times = []

        for i in range(iterations):
            start = time.time()

            splitter = DynamicTileSplit()
            tiles, tile_calc = splitter.split(
                test_img, tile_size, tile_size, overlap, mode
            )

            merger = DynamicTileMerge()
            result, = merger.merge(tiles, tile_calc, blend_width)

            # Ensure GPU operations complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        results[mode] = avg_time
        print(f"{mode:12s}: {avg_time*1000:.1f} ms")

    # Print relative performance
    baseline = results["linear"]
    print("\nRelative to linear:")
    for mode, time_val in results.items():
        overhead = (time_val / baseline - 1) * 100
        print(f"{mode:12s}: {overhead:+.1f}%")
    """


def main():
    """Run all visual tests."""
    print("="*60)
    print("Visual Integration Tests for Tile Blending")
    print("="*60)

    # Test pattern generation (always works)
    print("\n=== Testing Pattern Generation ===")
    output_dir = Path(__file__).parent / "visual_output"
    output_dir.mkdir(exist_ok=True)

    patterns = {
        "complex": create_test_pattern(512, 512),
        "gradient": create_gradient_pattern(512, 512),
        "texture": create_texture_pattern(512, 512),
    }

    for name, pattern in patterns.items():
        save_tensor_as_image(
            pattern,
            str(output_dir / f"pattern_{name}.png")
        )
    print(f"Test patterns saved to: {output_dir}")

    # Other tests (require implementation)
    test_full_pipeline_visual()
    test_seam_visibility()
    test_performance_benchmark()

    print("\n" + "="*60)
    print("Visual tests complete!")
    print("="*60)


if __name__ == "__main__":
    main()
