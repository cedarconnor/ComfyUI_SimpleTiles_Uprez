import sys
import os
from datetime import datetime
import re

import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import folder_paths

# Import blending module
try:
    from .blending import get_blender
except ImportError:
    from blending import get_blender


def order_by_center_last(tiles, image_width, image_height, tile_width, tile_height):
    # for 3x3: custom_order = [0, 2, 6, 8, 1, 3, 5, 7, 4] # First 4 corners, then the sides, then the center
    # order the tiles so they are add based on absolute distance from the center of the tile to the center of the image
    # this is done so that the center of the image is the last tile to be added, so that the center of the image is the most refined

    # get the center of the image
    center_x = image_width // 2
    center_y = image_height // 2

    # sort the tiles by distance from the center
    tiles = sorted(
        tiles,
        key=lambda tile: abs(tile[0] + tile_width // 2 - center_x)
        + abs(tile[1] + tile_height // 2 - center_y),
    )

    # reverse the order so that the center is last
    tiles = tiles[::-1]

    return tiles


def generate_tiles(
    image_width, image_height, tile_width, tile_height, overlap, offset=0
):
    tiles = []

    y = 0
    while y < image_height:
        if y == 0:
            next_y = y + tile_height - overlap + offset
        else:
            next_y = y + tile_height - overlap

        if y + tile_height >= image_height:
            y = max(image_height - tile_height, 0)
            next_y = image_height

        x = 0
        while x < image_width:
            if x == 0:
                next_x = x + tile_width - overlap + offset
            else:
                next_x = x + tile_width - overlap
            if x + tile_width >= image_width:
                x = max(image_width - tile_width, 0)
                next_x = image_width

            tiles.append((x, y))

            if next_x > image_width:
                break
            x = next_x

        if next_y > image_height:
            break
        y = next_y

    return order_by_center_last(tiles, image_width, image_height, tile_width, tile_height)


class DynamicTileSplit:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image tensor to split into tiles."}),
                "tile_width": ("INT", {"default": 512, "min": 1, "max": 10000, "tooltip": "Tile width in pixels before upscaling."}),
                "tile_height": ("INT", {"default": 512, "min": 1, "max": 10000, "tooltip": "Tile height in pixels before upscaling."}),
                "overlap": ("INT", {"default": 128, "min": 1, "max": 10000, "tooltip": "Overlap between adjacent tiles in pixels."}),
                "blend_mode": (["linear", "noise", "laplacian"], {"default": "linear", "tooltip": "Blending algorithm: linear (fast), noise (balanced), laplacian (best quality)."}),
                "offset": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Initial offset applied to the first tile row and column."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "TILE_CALC")
    FUNCTION = "process"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def process(self, image, tile_width, tile_height, overlap, blend_mode, offset):
        image_height = image.shape[1]
        image_width = image.shape[2]

        tile_coordinates = generate_tiles(
            image_width, image_height, tile_width, tile_height, overlap, offset
        )

        print("Tile coordinates: {}".format(tile_coordinates))
        print("Blend mode: {}".format(blend_mode))

        iteration = 1

        image_tiles = []
        tile_positions = []

        # Calculate grid dimensions
        rows = 0
        cols = 0
        for coord in tile_coordinates:
            x, y = coord
            row = y // (tile_height - overlap) if overlap < tile_height else 0
            col = x // (tile_width - overlap) if overlap < tile_width else 0
            rows = max(rows, row + 1)
            cols = max(cols, col + 1)

        for idx, tile_coordinate in enumerate(tile_coordinates):
            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))

            x_start = max(0, tile_coordinate[0])
            y_start = max(0, tile_coordinate[1])
            x_end = min(image_width, x_start + tile_width)
            y_end = min(image_height, y_start + tile_height)

            image_tile = image[
                :,
                y_start:y_end,
                x_start:x_end,
                :,
            ]

            image_tiles.append(image_tile)

            # Calculate row and column for this tile
            row = y_start // (tile_height - overlap) if overlap < tile_height else 0
            col = x_start // (tile_width - overlap) if overlap < tile_width else 0

            # Store tile position metadata
            tile_positions.append({
                "index": idx,
                "row": row,
                "col": col,
                "x1": x_start,
                "y1": y_start,
                "x2": x_end,
                "y2": y_end,
                "place_x": x_start,
                "place_y": y_start,
            })

            iteration += 1

        tiles_tensor = torch.stack(image_tiles).squeeze(1)

        # Extended tile_calc with new fields for advanced blending
        tile_calc = {
            "overlap": overlap,
            "overlap_x": overlap,
            "overlap_y": overlap,
            "image_height": image_height,
            "image_width": image_width,
            "offset": offset,
            "tile_height": tile_height,
            "tile_width": tile_width,
            "rows": rows,
            "cols": cols,
            "blend_mode": blend_mode,
            "tile_positions": tile_positions,
        }

        return (tiles_tensor, tile_calc)


class DynamicTileMerge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of tiles to merge back into the full image."}),
                "blend": ("INT", {"default": 64, "min": 0, "max": 4096, "tooltip": "Number of pixels to blend across tile edges."}),
                "tile_calc": ("TILE_CALC", {"tooltip": "Tile layout metadata returned by DynamicTileSplit."}),
            },
            "optional": {
                "auto_save": ("BOOLEAN", {"default": False, "tooltip": "Automatically save the merged result to the output directory."}),
                "filename_prefix": ("STRING", {"default": "tile_merge", "tooltip": "Filename prefix used when auto-saving the merged image."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "SimpleTiles Uprez/Dynamic"

    def process(self, images, blend, tile_calc, auto_save=False, filename_prefix="tile_merge"):
        filename_prefix = self._sanitize_prefix(filename_prefix)
        tile_height = images.shape[1]
        tile_width = images.shape[2]
        channels = images.shape[3]

        # Handle both old tuple format and new dict format for backwards compatibility
        if isinstance(tile_calc, dict):
            # New dict format
            overlap = tile_calc.get("overlap", 128)
            base_height = tile_calc.get("image_height", tile_height)
            base_width = tile_calc.get("image_width", tile_width)
            offset = tile_calc.get("offset", 0)
            base_tile_height = tile_calc.get("tile_height", tile_height)
            base_tile_width = tile_calc.get("tile_width", tile_width)
            blend_mode = tile_calc.get("blend_mode", "linear")
            tile_positions = tile_calc.get("tile_positions", None)
            rows = tile_calc.get("rows", 1)
            cols = tile_calc.get("cols", 1)
        elif isinstance(tile_calc, tuple):
            # Old tuple format - backwards compatibility
            if len(tile_calc) >= 6:
                overlap, base_height, base_width, offset, base_tile_height, base_tile_width = tile_calc[:6]
            else:
                overlap, base_height, base_width, offset = tile_calc
                base_tile_height = tile_height
                base_tile_width = tile_width
            blend_mode = "linear"  # Default to linear for old workflows
            tile_positions = None
            rows = 1
            cols = 1
        else:
            raise ValueError(f"Unexpected tile_calc type: {type(tile_calc)}")

        scale_y = tile_height / base_tile_height if base_tile_height else 1.0
        scale_x = tile_width / base_tile_width if base_tile_width else 1.0
        average_scale = (scale_x + scale_y) / 2.0

        scaled_final_height = max(tile_height, int(round(base_height * scale_y)))
        scaled_final_width = max(tile_width, int(round(base_width * scale_x)))
        scaled_overlap = max(0, int(round(overlap * average_scale)))
        scaled_offset = int(round(offset * average_scale))

        print("Tile height (scaled): {}".format(tile_height))
        print("Tile width (scaled): {}".format(tile_width))
        print("Base height: {} -> scaled height: {}".format(base_height, scaled_final_height))
        print("Base width: {} -> scaled width: {}".format(base_width, scaled_final_width))
        print("Overlap: {} -> scaled overlap: {}".format(overlap, scaled_overlap))
        print("Blend mode: {}".format(blend_mode))

        tile_coordinates = generate_tiles(
            scaled_final_width, scaled_final_height, tile_width, tile_height, scaled_overlap, scaled_offset
        )

        print("Tile coordinates: {}".format(tile_coordinates))

        # Initialize canvas
        canvas = torch.zeros((1, scaled_final_height, scaled_final_width, channels), dtype=images.dtype, device=images.device)

        # Get appropriate blender
        try:
            blender = get_blender(blend_mode, blend)
            use_advanced_blending = True
            print("Using {} blender with blend width {}".format(blend_mode, blend))
        except Exception as e:
            print(f"Warning: Could not initialize {blend_mode} blender ({e}), falling back to legacy blending")
            use_advanced_blending = False

        # Process each tile
        index = 0
        iteration = 1

        for tile_coordinate in tile_coordinates:
            # Bounds check - ensure we don't exceed available tiles
            if index >= len(images):
                print(f"Warning: More coordinates ({len(tile_coordinates)}) than tiles ({len(images)}). Stopping at tile {index}.")
                break

            image_tile = images[index:index+1]  # Keep batch dimension
            x = tile_coordinate[0]
            y = tile_coordinate[1]

            print("Processing tile {} of {}".format(iteration, len(tile_coordinates)))
            print("Tile coordinate: {}".format(tile_coordinate))

            if use_advanced_blending and tile_positions and index < len(tile_positions):
                # Use new blender system with position metadata
                position = tile_positions[index]

                # Scale the placement coordinates
                position_scaled = {
                    "index": position["index"],
                    "row": position["row"],
                    "col": position["col"],
                    "place_x": int(position["place_x"] * scale_x),
                    "place_y": int(position["place_y"] * scale_y),
                }

                # Update tile_calc for blender
                tile_calc_for_blender = {
                    "image_height": scaled_final_height,
                    "image_width": scaled_final_width,
                    "overlap": scaled_overlap,
                    "rows": rows,
                    "cols": cols,
                }

                canvas = blender.blend_tiles(canvas, image_tile, position_scaled, tile_calc_for_blender)

            else:
                # Fallback to legacy blending (old behavior)
                weight_matrix = torch.ones((tile_height, tile_width, channels), device=images.device)

                # Blend border
                for i in range(blend):
                    weight = float(i) / blend
                    weight_matrix[i, :, :] *= weight  # Top rows
                    weight_matrix[-(i + 1), :, :] *= weight  # Bottom rows
                    weight_matrix[:, i, :] *= weight  # Left columns
                    weight_matrix[:, -(i + 1), :] *= weight  # Right columns

                # Only blend with already processed pixels
                old_tile = canvas[:, y:y+tile_height, x:x+tile_width, :]
                old_tile_count = (old_tile.abs().sum(dim=-1, keepdim=True) > 0).float()

                weight_matrix = (
                    weight_matrix * (old_tile_count != 0).float()
                    + (old_tile_count == 0).float()
                )

                blended = image_tile[0] * weight_matrix + old_tile[0] * (1 - weight_matrix)
                canvas[:, y:y+tile_height, x:x+tile_width, :] = blended.unsqueeze(0)

            index += 1
            iteration += 1

        saved_path = None
        if auto_save:
            try:
                saved_path = self._save_merged_image(canvas, filename_prefix)
            except Exception as exc:
                print(f"SimpleTilesUprezDynamicTileMerge: Failed to save merged image ({exc})")
        if saved_path:
            print(f"SimpleTilesUprezDynamicTileMerge: Saved merged image to {saved_path}")

        return [canvas]

    @staticmethod
    def _sanitize_prefix(name: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
        return sanitized or "tile_merge"

    def _save_merged_image(self, tensor: torch.Tensor, prefix: str) -> str:
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        full_path = os.path.join(output_dir, filename)

        image_tensor = tensor[0].clamp(0, 1)
        image_array = (
            image_tensor.mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        Image.fromarray(image_array).save(full_path)
        return full_path


NODE_CLASS_MAPPINGS = {
    "DynamicTileSplit": DynamicTileSplit,
    "DynamicTileMerge": DynamicTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynamicTileSplit": "DynamicTileSplit",
    "DynamicTileMerge": "DynamicTileMerge",
}





