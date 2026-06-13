# MPV GLSL Shader → WebGPU WGSL Porting Guide

## Project Overview

Porting [RAVU-Zoom-AR](https://github.com/bjin/mpv-prescalers) (mpv hook shader) to a browser-based WebGPU Tampermonkey userscript. Two variants:

- **r2**: 4×4 sampling window (16 samples), 2-column LUT (x ∈ {0.0, 0.5}), fixed 2× scale
- **r3**: 6×6 sampling window (36 samples), 5-column LUT (x ∈ {0.0, 0.2, 0.4, 0.6, 0.8}), arbitrary ratio scale

## GLSL → WGSL Mapping

| GLSL | WGSL | Notes |
|------|------|-------|
| `mix(a,b,c)` | `mix(a,b,c)` | Same |
| `dot(a,b)` | `dot(a,b)` | Same |
| `floor(x)` | `floor(x)` | Same |
| `fract(x)` | `fract(x)` | Same |
| `atan(y,x)` | `atan2(y,x)` | WGSL uses `atan2`, GLSL uses `atan` with 2 args |
| `mod(x,y)` | `x - y * floor(x / y)` | No built-in `mod`; implement manually |
| `mix(a,b,c)` where `c` is bool | `select(b,a,c)` | WGSL has no bool `mix`; use `select` |
| `texture(sampler, coord)` | `textureSample(tex, sampler, coord)` | Different API |
| `vec3` | `vec3<f32>` | Explicit type param required |
| `mat4x3` | `mat4x3<f32>` | Explicit type param required |
| `matrixCompMult(a,b)` | Custom fn (see below) | Not built-in |
| `vec2(0.5)` | `vec2<f32>(0.5)` | Explicit type |

## matrixCompMult (needed by AR)

```wgsl
fn matrixCompMult(a: mat4x3<f32>, b: mat4x3<f32>) -> mat4x3<f32> {
    return mat4x3<f32>(a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]);
}
```

## Struct syntax difference

GLSL uses `,` between fields; WGSL uses `;`:

```wgsl
struct Params {
    inputSize: vec2<f32>,
    outputSize: vec2<f32>,    // ← comma OK in WGSL (like GLSL)
};
```

WGSL does accept commas between fields (with `@location` annotations), but semicolons also work.

## LUT Data Loading

The hook file stores LUT weight data as raw hex in `//!TEXTURE ... //!SIZE W H //!FORMAT rgba16f` blocks.

### Critical: Hex format

**The hex string is raw rgba32f bytes** despite the hook header saying `rgba16f` — each pixel = 4 float32 values (32 hex chars). The hook comment `//!FORMAT rgba16f` is the shader's target format, not the storage format in the file.

### Heap issue

Pasting the hex string directly into source code creates huge string literals:
- r2 LUT2: 1,492,992 chars (18 × 2592 × 4 × 8 hex chars)
- r2 LUT2_AR: same (also 18 × 2592)
- r3 LUT3: 3,732,480 chars (45 × 2592 × 4 × 8)
- r3 LUT3_AR: 1,492,992 chars (18 × 2592 × 4 × 8)

Tampermonkey handles this, but DevTools becomes nearly unusable after install.

### Conversion pipeline

```
hex string → hexToUint8Array() → Float32Array (rgba32f) → float32ToHalf() → Uint8Array (rgba16f) → device.queue.writeTexture()
```

1. **hexToUint8Array**: strip whitespace, parse pairs to bytes
2. **Float32Array overlay**: treat raw bytes as float32 values
3. **float32ToHalf**: custom IEEE 754 float32 → float16 converter
4. **Align rows**: WebGPU requires `bytesPerRow` multiple of 256
5. **writeTexture**: upload to `rgba16float` texture

### float32ToHalf implementation

```js
function float32ToHalf(v) {
    const buf = new ArrayBuffer(4);
    new Float32Array(buf)[0] = v;
    const u = new Uint32Array(buf)[0];
    const s = (u >> 16) & 0x8000;
    let e = ((u >> 23) & 0xff) - 112;
    let m = (u >> 13) & 0x3ff;
    if (e <= 0) { m = (m | 0x400) >> (1 - e); e = 0; }
    else if (e > 30) { e = 0x1f; m = 0; }
    return s | (e << 10) | m;
}
```

### Why not rgba32float texture?

WebGPU restricts `rgba32float` textures to `UnfilterableFloat` sampler, which means `nearest` filtering only. RAVU-Zoom requires `linear` filtering for the LUT — hence the CPU-side half-float conversion.

## Edge Tensor (Structure Tensor)

Both r2 and r3 compute 16 gradient pairs `(gx, gy)` from luma values, each with a specific weight:

```
abd += vec3(gx * gx, gx * gy, gy * gy) * weight
```

The structure tensor `[[a,b],[b,d]]` eigenvalues determine edge properties:

```wgsl
let T = a + d;
let D = a * d - b * b;
let delta = sqrt(max(T * T / 4.0 - D, 0.0));
let L1 = T / 2.0 + delta;
let L2 = T / 2.0 - delta;
let sqrtL1 = sqrt(max(L1, 0.0));  // WGSL: sqrt of negative → NaN, use max
let sqrtL2 = sqrt(max(L2, 0.0));
```

**Important**: WGSL `sqrt` of negative returns NaN (unlike GLSL which may clamp). Use `max(v, 0.0)` before `sqrt`.

Edge direction, strength, and coherence are quantized → 288 possible `coord_y` values:

```wgsl
let angle = floor(theta * 24.0 / 3.141592653589793);
let strength = select(select(0.0, 1.0, lambda >= 0.004), select(2.0, 3.0, lambda >= 0.05), lambda >= 0.016);
let coherence = select(select(0.0, 1.0, mu >= 0.25), 2.0, mu >= 0.5);
let coord_y = ((angle * 4.0 + strength) * 3.0 + coherence) / 288.0;
```

Note use of `select()` instead of nested `mix()` for boolean conditions.

## LUT Weighted Upscaling (Forward + Inverse)

The LUT is sampled at sub-pixel positions with multiple x-column groups:

**r2** (2 groups):
```
x = 0.0: samples 0,1,2,3      → col 0-3
x = 0.5: samples 4,5,6,7      → col 0-3
x = 0.0 (inv): samples 15,14,13,12
x = 0.5 (inv): samples 11,10,9,8
```

**r3** (5 groups):
```
x = 0.0: samples 0,1,2,3      → col 0-3
x = 0.2: samples 4,5,6,7      → col 0-3
x = 0.4: samples 8,9,10,11    → col 0-3
x = 0.6: samples 12,13,14,15  → col 0-3
x = 0.8: samples 16,17        → col 0-1 (only 2)
... plus mirrored inverse side ...
```

The last column in r3's forward path uses only 2 weights (samples 16,17) — r3 has 5 groups × 4 weights = 20 max but total 18 samples per side. Similarly inverse side: 18 samples in ranges reverse order.

## Anti-Ringing (AR)

AR uses a separate LUT (`ravu_zoom_lut2_ar` / `ravu_zoom_lut3_ar`, always 18×2592) with a different sub-pix scale:

```wgsl
let subpix_ar = subpix / vec2<f32>(2.0, 288.0);           // r3 AR
let subpix    = subpix / vec2<f32>(5.0, 288.0);           // r3 LUT
```

AR applies a steep sigmoid via 5 repeated `matrixCompMult(cg, cg)` calls, computing dynamic `lo`/`hi` clamping bounds:

```wgsl
cg = mat4x3<f32>(0.1 + sample7, 1.1 - sample7, 0.1 + sample8, 1.1 - sample8);
cg1 = cg;
cg = matrixCompMult(cg, cg); cg = matrixCompMult(cg, cg);
cg = matrixCompMult(cg, cg); cg = matrixCompMult(cg, cg);
cg = matrixCompMult(cg, cg);
hi += cg[0] * w[0] + cg[2] * w[1];
lo += cg[1] * w[0] + cg[3] * w[1];
cg = matrixCompMult(cg, cg1);     // undo 1 power for hi2/lo2
hi2 += cg[0] * w[0] + cg[2] * w[1];
lo2 += cg[1] * w[0] + cg[3] * w[1];
```

Final: `res = mix(res, clamp(res, lo, hi), 0.8)`

## WebGPU Pipeline Setup

```js
const context = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: presentationFormat, alphaMode: 'premultiplied' });

// render pipeline
const pipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shaderModule, entryPoint: 'vs_main' },
    fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format: presentationFormat }] },
    primitive: { topology: 'triangle-list' },
});
```

### Bind group layout (WGSL → JS)

| Binding | WGSL | JS |
|---------|------|-----|
| 0 | `var<uniform> params: Params` | `{ buffer: uniformBuffer }` |
| 1 | `var inputSampler: sampler` | `inputSampler` |
| 2 | `var inputTexture: texture_2d<f32>` | `videoTexture.createView()` |
| 3 | `var lutSampler: sampler` | `lutSampler` |
| 4 | `var lutTexture: texture_2d<f32>` | `lutTexture.createView()` |
| 5 | `var lutArSampler: sampler` | `lutArSampler` |
| 6 | `var lutArTexture: texture_2d<f32>` | `lutArTexture.createView()` |

### Uniform buffer (32 bytes = 8 × float32)

```js
const uniformData = new Float32Array([
    video.videoWidth, video.videoHeight,   // inputSize
    targetWidth, targetHeight,             // outputSize
    1.0 / video.videoWidth, 1.0 / video.videoHeight,  // texelSize
    0.0, 0.0                               // padding
]);
```

### Video frame capture

```js
const videoTexture = device.createTexture({
    size: [video.videoWidth, video.videoHeight, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
});
device.queue.copyExternalImageToTexture({ source: video }, { texture: videoTexture }, [video.videoWidth, video.videoHeight]);
```

## Render Target Sizing

**r2**: fixed 2× scale — `canvas.width = video.videoWidth * 2`

**r3**: arbitrary ratio — `canvas.width = Math.round(video.offsetWidth * devicePixelRatio)`

This avoids the browser's built-in downscaling step, giving the shader full control over the output resolution. The canvas CSS size matches `video.getBoundingClientRect()` with aspect-ratio-aware centering.

## Canvas Positioning Over Video

```js
canvas.style.position = 'absolute';
canvas.style.zIndex = getComputedStyle(video).zIndex;  // match video stacking
video.parentNode.insertBefore(canvas, video.nextSibling);
video.style.opacity = '0';   // hide original video
```

Positioning calculation (aspect-ratio aware):

```js
const vr = video.getBoundingClientRect();
// find containing block (first non-static parent)
let cr = canvas.parentNode;
while (cr && cr !== document.documentElement) {
    if (getComputedStyle(cr).position !== 'static') break;
    cr = cr.parentNode;
}
const cb = cr.getBoundingClientRect();
// letterbox/pillarbox centering
if (Math.abs(vr.width / vr.height - ar) > 0.001) {
    // adjust width or height to match video aspect ratio
}
canvas.style.left = (vr.left - cb.left) + 'px';
canvas.style.top = (vr.top - cb.top) + 'px';
```

### Fullscreen support

```js
document.addEventListener('fullscreenchange', () => {
    const fs = document.fullscreenElement;
    if (fs && fs !== document.body && !fs.contains(canvas)) {
        fs.appendChild(canvas);   // reparent into fullscreen element
    } else if (!fs && canvas.parentNode !== document.body) {
        document.body.appendChild(canvas);
    }
});
```

## Edge Cases & Gotchas

1. **NaN from sqrt**: WGSL doesn't clamp negative sqrt → use `max(v, 0.0)` before `sqrt`
2. **atan2 args**: WGSL `atan2(y, x)` matches GLSL `atan(y, x)` but differs from GLSL `atan(y_over_x)`
3. **select for bool mix**: WGSL `select(b, a, cond)` = GLSL `mix(a, b, cond)` when cond is bool
4. **mod implementation**: `x - y * floor(x / y)` works, watch for floating errors near 0
5. **Texture row alignment**: WebGPU `writeTexture` requires `bytesPerRow` multiple of 256
6. **videoTexture recreation per frame**: Must destroy old texture before creating new one to avoid memory leak
7. **canvas re-parenting in fullscreen**: When entering fullscreen, canvas must be a child of the fullscreen element or it won't be visible
8. **LUT hex size**: Ensure hex string length matches expected dimensions — wrong length → Float32Array length mismatch → silent wrong data or crash
9. **devicePixelRatio changes**: Monitor via `matchMedia` listener if window moves between displays
10. **sigmoid power iteration**: The 5× `matrixCompMult(cg, cg)` must be exact — even one fewer changes the anti-ringing behavior noticeably

## Testing

- Load via Tampermonkey on B站 or any H.264/H.265 video page
- Toggle with floating RAVU/OFF button
- Check console for WebGPU errors via `uncapturederror` event
- Compare with original mpv shader output (same RAVU-Zoom-AR algorithm)
