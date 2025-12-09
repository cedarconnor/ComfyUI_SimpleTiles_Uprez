
# ComfyUI_SimpleTiles_Uprez

## DynamicTileSplit / DynamicTileMerge
![](dynamic.png)

Splits image into tiles for processing (e.g., upscaling), then merges them back seamlessly using advanced blending algorithms.

### Overview

**DynamicTileSplit** automatically splits an image into overlapping tiles based on image size and desired tile dimensions. Tiles can have different aspect ratios than the source image.

**DynamicTileMerge** reconstructs the full image from processed tiles using one of three blending modes to eliminate visible seams.

### Parameters

**Tile Width / Height**: Size of each tile before processing (e.g., 512×512 for models that work best at that resolution)

**Overlap**: How many pixels adjacent tiles overlap. Higher overlap = better blending quality but more processing. Recommended: 64-128 pixels.

**Blend Mode**: Algorithm used to merge overlapping regions (see detailed comparison below)

**Blend Width**: How many pixels over which blending occurs. Must be ≤ overlap. Higher values = smoother transitions.

**Offset**: Initial offset for first tile position (advanced use)

---

## Blend Modes - Detailed Comparison

ComfyUI_SimpleTiles_Uprez offers three blending algorithms, each with different quality/speed trade-offs:

### 🔹 Linear Blending (Default)
**Speed: ⚡⚡⚡ Fastest | Quality: ⭐⭐ Good | Best for: Quick iterations**

#### How It Works
Linear blending creates smooth transitions by gradually fading between tiles using simple gradients:
- Each tile edge gets a linear weight ramp from 0→1
- Overlapping pixels are blended as: `result = tile1 × (1-weight) + tile2 × weight`
- Weights are applied independently to each edge (top, bottom, left, right)

#### Algorithm
1. Create a weight matrix for each tile (all 1s initially)
2. Apply linear gradients at borders:
   - Top edge: weights from 0→1 over blend_width pixels
   - Bottom edge: weights from 1→0 over blend_width pixels
   - Left/right edges: same principle
3. Blend overlapping regions using weighted average
4. Place tiles on canvas in center-last order (edges first, center last)

#### Pros & Cons
✅ **Pros:**
- Fastest method (~2ms per tile)
- Low memory usage
- Predictable, consistent results
- Good for most natural images

❌ **Cons:**
- Visible seams on high-frequency details (text, fine patterns)
- Can create ghosting artifacts on sharp edges
- Brightness variations may be visible on gradients

#### Best Used For
- Quick previews and iterations
- Natural photos without fine details
- When processing speed is critical
- Images with soft textures (sky, skin, foliage)

---

### 🔸 Noise Blending
**Speed: ⚡⚡ Fast | Quality: ⭐⭐⭐⭐ Excellent | Best for: Most production work**

#### How It Works
Noise blending uses Perlin noise to create organic, irregular boundaries instead of straight lines:
- Generates 1D Perlin noise patterns along tile edges
- Uses Fractional Brownian Motion (FBM) for natural randomness
- Creates a wavy transition zone that mimics natural irregularities
- Sigmoid smoothing ensures smooth falloff

#### Algorithm
1. **Generate Perlin noise seam:**
   - Create 1D Perlin noise using smoothstep interpolation
   - Apply FBM (multiple octaves): `Σ amplitude × perlin(frequency × x)`
   - Default: 3 octaves, frequency 0.05, each octave halves amplitude

2. **Create blend mask:**
   - Base linear gradient (0→1 across blend_width)
   - Perturb with noise: `position = linear + noise × displacement`
   - Apply sigmoid smoothing: `sigmoid(10 × (x - 0.5))` for smooth transitions

3. **Blend using noisy boundary:**
   - Mask becomes irregular instead of straight line
   - Visual discontinuities are broken up and hidden
   - Human eye doesn't perceive irregular boundaries as easily

#### Pros & Cons
✅ **Pros:**
- Eliminates visible seams in most cases
- Handles high-frequency details well
- Natural-looking transitions
- 7× slower than linear (still fast: ~14ms per tile)
- Excellent quality/speed balance

❌ **Cons:**
- Slight randomness (different seeds = different seams)
- Can create minor texture variations in flat areas
- Not deterministic (though reproducible with fixed seed)

#### Best Used For
- **Recommended for most production work**
- Images with detailed textures (fabric, foliage, architecture)
- When seams must be invisible
- Upscaling photos with mixed content
- When you need better quality than linear without laplacian's overhead

#### Configuration
- `frequency`: Controls noise scale (0.05 = gentle waves, 0.2 = tight ripples)
- `octaves`: Detail levels (3 = balanced, 5 = more complex patterns)
- Higher octaves add finer detail but slower processing

---

### 🔶 Laplacian Pyramid Blending
**Speed: ⚡ Slower | Quality: ⭐⭐⭐⭐⭐ Maximum | Best for: Final high-quality output**

#### How It Works
Laplacian pyramid blending is a sophisticated multi-band algorithm that blends different frequency bands separately:
- Decomposes images into multiple frequency levels (like low-pass filters)
- Blends low frequencies over wide areas, high frequencies over narrow areas
- Reconstructs the final image by recombining all frequency bands
- Based on Burt & Adelson's seminal 1983 paper

#### Algorithm
1. **Build Gaussian Pyramids:**
   - Repeatedly blur and downsample each tile
   - Level 0: Original (128×128)
   - Level 1: Downsampled (64×64)
   - Level 2: Downsampled (32×32)
   - Level 3: Downsampled (16×16)
   - Each level is half the size of the previous

2. **Create Laplacian Pyramids:**
   - For each level: `laplacian[i] = gaussian[i] - upsample(gaussian[i+1])`
   - Extracts detail/high-frequency information at each scale
   - Lowest level (residual) stays as-is

3. **Build Mask Pyramid:**
   - Create Gaussian pyramid of the blend mask
   - Lower levels = blurrier masks = wider blend zones for low frequencies

4. **Blend Each Level Separately:**
   - `blended[i] = background[i] × (1 - mask[i]) + foreground[i] × mask[i]`
   - High-frequency details blend over narrow zones
   - Low-frequency colors blend over wide zones
   - This is the key: different frequencies blend at different scales

5. **Reconstruct Final Image:**
   - Collapse pyramid: start with lowest resolution
   - Progressively upsample and add detail layers
   - `img = upsample(img) + laplacian[i]` for each level

#### Why This Works
- High frequencies (edges, textures) blend over few pixels → preserves sharpness
- Low frequencies (colors, lighting) blend over many pixels → smooth transitions
- Mimics how human visual system processes different spatial frequencies
- Prevents "double edges" and ghosting artifacts

#### Pros & Cons
✅ **Pros:**
- **Best possible quality** - imperceptible seams even on challenging content
- Handles sharp edges perfectly (no ghosting)
- Preserves fine details while smoothing colors
- Excellent for text, patterns, architectural details
- Scientifically proven algorithm used in professional tools

❌ **Cons:**
- 22× slower than linear (~44ms per tile)
- Higher memory usage (stores multiple pyramid levels)
- Slight computation overhead for small images
- Overkill for simple natural photos

#### Best Used For
- **Final production renders** where quality is paramount
- Images with sharp edges and fine details
- Text overlays or UI elements
- Architectural photography with hard lines
- When you need absolutely invisible seams
- High-resolution outputs where artifacts would be noticeable

#### Configuration
- `levels`: Pyramid depth (4 = default, 6 = finer decomposition)
- More levels = better quality but slower processing
- Automatically clamps output to [0, 1] to prevent reconstruction artifacts

---

## Blend Mode Selection Guide

### Quick Reference

| Content Type | Recommended Mode | Why |
|--------------|------------------|-----|
| Natural photos (landscapes, portraits) | **Linear** or **Noise** | Fast, good enough for organic content |
| Detailed textures (fabric, foliage) | **Noise** | Hides seams in complex patterns |
| Architecture, buildings | **Noise** or **Laplacian** | Handles geometric patterns well |
| Text, UI elements, graphics | **Laplacian** | Preserves sharp edges perfectly |
| Mixed content (photo + graphics) | **Laplacian** | Best overall quality |
| Quick preview/iteration | **Linear** | Fastest processing |
| Final production output | **Laplacian** | Maximum quality |

### Performance Comparison
*(Tested on 512×512 tiles, 64px overlap, 32px blend width)*

| Mode | Speed | Relative | Quality | Memory |
|------|-------|----------|---------|--------|
| Linear | ~2ms/tile | 1× (baseline) | ⭐⭐ | Low |
| Noise | ~14ms/tile | 7× slower | ⭐⭐⭐⭐ | Low |
| Laplacian | ~44ms/tile | 22× slower | ⭐⭐⭐⭐⭐ | Medium |

### Recommendations

**For most users:** Start with **Noise** blending. It offers excellent quality with minimal performance impact and handles 95% of use cases perfectly.

**Use Linear when:**
- You're iterating quickly and need fast previews
- Processing hundreds of images in batch
- Image content is simple/natural without fine details

**Use Laplacian when:**
- Quality is paramount and time doesn't matter
- Image contains text, logos, or UI elements
- You notice visible seams with other modes
- Creating final production output for clients/portfolio

---

## Advanced Usage Tips

### Overlap Recommendations
- **Minimum:** 32 pixels (visible seams likely)
- **Recommended:** 64-128 pixels (good balance)
- **Maximum quality:** 128-256 pixels (diminishing returns)

### Blend Width Guidelines
- Set to 50-80% of overlap value
- Too low: abrupt transitions still visible
- Too high: can exceed overlap (automatically clamped)
- Linear: can use smaller blend width
- Laplacian: can use larger blend width effectively

### Backwards Compatibility
The nodes are fully backwards compatible with existing workflows:
- Old workflows without `blend_mode` automatically use Linear
- Old `tile_calc` tuple format is supported
- No changes needed to existing saved workflows


## Legacy
DynamicTileSplit and DynamicTileMerge are the new versions of TileSplit and TileMerge. They are more flexible and easier to use.

Legacy nodes don't work well if image ratio and tile ratio is different. 

Use TileCalc to calculate the final image size, pipe the final size to TileMerge and ImageScale.


### TileSplit (Legacy)
Splits image into tiles. Overlap value decides how much overlap there is between tiles on y axis, x axis is calculated to have the same ratio to image height as y axis.

### TileMerge (Legacy)
Merge tiles into image. 

**Overlap** value decides how much overlap there is between tiles on y axis, x axis is calculated to have the same ratio to image height as y axis. Should be set to same value as used in TileSplit.

**Blend** value decides how many pixels the blending is done over. Should be less than overlap value. Blending is done linearly from 0 to 1 over the blend distance.

### TileCalc (Legacy)

Util to calculate final image size based on tile sizes and overlaps. 



## Example Ipadapter
![](ipadapter.png)

