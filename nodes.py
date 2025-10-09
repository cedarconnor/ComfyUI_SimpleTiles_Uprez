from .standard import TileSplit, TileMerge, TileCalc
from .dynamic import DynamicTileSplit, DynamicTileMerge

# Register nodes under unique names to avoid clashes with the original SimpleTiles pack.
NODE_CLASS_MAPPINGS = {
    "SimpleTilesUprezTileSplit": TileSplit,
    "SimpleTilesUprezTileMerge": TileMerge,
    "SimpleTilesUprezTileCalc": TileCalc,
    "SimpleTilesUprezDynamicTileSplit": DynamicTileSplit,
    "SimpleTilesUprezDynamicTileMerge": DynamicTileMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleTilesUprezTileSplit": "TileSplit (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezTileMerge": "TileMerge (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezTileCalc": "TileCalc (SimpleTiles Uprez Legacy)",
    "SimpleTilesUprezDynamicTileSplit": "TileSplit (SimpleTiles Uprez Dynamic)",
    "SimpleTilesUprezDynamicTileMerge": "TileMerge (SimpleTiles Uprez Dynamic)",
}
