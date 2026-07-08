// ==UserScript==
// @name         SSimDownscaler & SSimSuperRes for Bilibili & Youtube
// @namespace    http://tampermonkey.net/
// @version      0.1.0
// @description  SSimDownscaler + SSimSuperRes (WebGPU port) for Bilibili & Youtube
// @author       Ckrvxr,igv
// @match        *://*.bilibili.com/*
// @match        *://*.youtube.com/*
// @license      LGPL-3.0-or-later
// @grant        none
// ==/UserScript==

(async function() {
    'use strict';

    if (window.top !== window.self) return;

    if (!navigator.gpu) {
        console.error("[SSim] WebGPU not supported");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    console.log("[SSim] WebGPU adapter:", adapter.info ? `${adapter.info.vendor} ${adapter.info.architecture}` : "unknown");
    const device = await adapter.requestDevice();
    console.log("[SSim] WebGPU device ready, features:", [...device.features].join(", "));

    let hasWebGPUError = false;
    let lastErrorTime = 0;
    device.addEventListener('uncapturederror', (event) => {
        console.error("[SSim] WebGPU error:", event.error.message);
        hasWebGPUError = true;
        lastErrorTime = Date.now();
    });

    let activeState = null;
    let hijackSeq = 0;

    const RG16F = 'rg16float';
    const RGBA16F = 'rgba16float';

    function cleanupActive(restoreOpacity) {
        if (activeState) {
            activeState.running = false;
            if (restoreOpacity && activeState.video?.isConnected) {
                activeState.video.style.opacity = '1';
            }
            activeState.canvas?.remove();
            activeState = null;
            console.log("[SSim] Cleaned up stale render state");
        }
    }

    const wgslCode = `
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Params {
    videoSize: vec2<f32>,
    videoTexelSize: vec2<f32>,
    canvasSize: vec2<f32>,
    canvasTexelSize: vec2<f32>,
    inputTexSize: vec2<f32>,
    inputTexTexelSize: vec2<f32>,
    _pad1: vec2<f32>,
    _pad2: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var texA: texture_2d<f32>;
@group(0) @binding(3) var texB: texture_2d<f32>;
@group(0) @binding(4) var texC: texture_2d<f32>;
@group(0) @binding(5) var texD: texture_2d<f32>;

const color_primary = vec3<f32>(0.2126, 0.7152, 0.0722);
const PI = 3.141592653589793;

fn Luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, color_primary);
}

fn LumaSq(rgb: vec3<f32>) -> f32 {
    return dot(rgb * rgb, color_primary);
}

fn MN(B: f32, C: f32, x: f32) -> f32 {
    if (x < 1.0) {
        return ((2.0 - 1.5 * B - C) * x + (-3.0 + 2.0 * B + C)) * x * x + (1.0 - B / 3.0);
    }
    return (((-B / 6.0 - C) * x + (B + 5.0 * C)) * x + (-2.0 * B - 8.0 * C)) * x + (4.0 / 3.0 * B + 4.0 * C);
}

fn KernelDown(x: f32) -> f32 {
    return MN(0.0, 0.5, abs(x));
}

fn KernelSR(x: f32) -> f32 {
    return MN(0.334, 0.333, abs(x));
}

fn KernelExp(x: f32) -> f32 {
    return pow(1.0 / 2.0, abs(x));
}

fn KernelHann(x: f32) -> f32 {
    return cos(PI * x / 3.0);
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var pos = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
    );
    var uv = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0)
    );
    var output: VertexOutput;
    output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
    output.uv = uv[vertexIndex];
    return output;
}

@fragment
fn fs_bilinear(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSampleLevel(texA, inputSampler, input.uv, 0.0);
}

@fragment
fn fs_down_l2_v(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let axis: i32 = 1;
    let taps = 2.0;

    let base = uv;
    let low_v = ceil((uv - vec2<f32>(taps) * params.canvasTexelSize) * params.videoSize - vec2<f32>(0.5));
    let high_v = floor((uv + vec2<f32>(taps) * params.canvasTexelSize) * params.videoSize - vec2<f32>(0.5));

    var W = 0.0;
    var avg = vec4<f32>(0.0);
    var pos = base;

    let lo = low_v[axis];
    let hi = high_v[axis];

    var k: f32 = lo;
    while (k <= hi) {
        pos[axis] = params.videoTexelSize[axis] * (k + 0.5);
        let rel = (pos[axis] - base[axis]) * params.canvasSize[axis];
        let w = KernelDown(rel);
        let tex = textureSampleLevel(texA, inputSampler, pos, 0.0);
        avg = avg + w * tex * tex;
        W = W + w;
        k = k + 1.0;
    }
    avg = avg / W;
    return avg;
}

@fragment
fn fs_down_l2_h(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let axis: i32 = 0;
    let taps = 2.0;

    let base = uv;
    let low_v = ceil((uv - vec2<f32>(taps) * params.canvasTexelSize) * params.inputTexSize - vec2<f32>(0.5));
    let high_v = floor((uv + vec2<f32>(taps) * params.canvasTexelSize) * params.inputTexSize - vec2<f32>(0.5));

    var W = 0.0;
    var avg = vec4<f32>(0.0);
    var pos = base;

    let lo = low_v[axis];
    let hi = high_v[axis];

    var k: f32 = lo;
    while (k <= hi) {
        pos[axis] = params.inputTexTexelSize[axis] * (k + 0.5);
        let rel = (pos[axis] - base[axis]) * params.canvasSize[axis];
        let w = KernelDown(rel);
        let tex = textureSampleLevel(texA, inputSampler, pos, 0.0);
        avg = avg + w * tex;
        W = W + w;
        k = k + 1.0;
    }
    avg = avg / W;
    return avg;
}

@fragment
fn fs_down_mr(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let oversharp = 0.0;
    let sigma_nsq = 10.0 / (255.0 * 255.0);
    let taps = 3.0;

    let low_x = ceil(-0.5 * taps - 0.5);
    let high_x = floor(0.5 * taps - 0.5);
    let low_y = ceil(-0.5 * taps - 0.5);
    let high_y = floor(0.5 * taps - 0.5);

    let zero3 = vec3<f32>(0.0);
    var avg = mat3x3<f32>(zero3, zero3, zero3);
    var W = 0.0;
    var pos = uv;

    var j: f32 = low_y;
    while (j <= high_y) {
        pos[1] = uv[1] + params.canvasTexelSize[1] * j;
        let rel_y = j;
        let wy = KernelExp(abs(rel_y));

        var row = mat3x3<f32>(zero3, zero3, zero3);
        var rowW = 0.0;
        var pos2 = pos;

        var i: f32 = low_x;
        while (i <= high_x) {
            pos2[0] = uv[0] + params.canvasTexelSize[0] * i;
            let rel_x = i;
            let wx = KernelExp(abs(rel_x));

            let L = textureSampleLevel(texA, inputSampler, pos2, 0.0).rgb;
            let L2rgb = textureSampleLevel(texB, inputSampler, pos2, 0.0).rgb;
            row = row + wx * mat3x3<f32>(L, L * L, L2rgb);
            rowW = rowW + wx;
            i = i + 1.0;
        }
        row = mat3x3<f32>(row[0] / rowW, row[1] / rowW, row[2] / rowW);
        avg = avg + wy * row;
        W = W + wy;
        j = j + 1.0;
    }
    avg = mat3x3<f32>(avg[0] / W, avg[1] / W, avg[2] / W);

    let Sl = Luma(max(avg[1] - avg[0] * avg[0], vec3<f32>(0.0)));
    let Sh = Luma(max(avg[2] - avg[0] * avg[0], vec3<f32>(0.0)));
    let R = mix(sqrt((Sh + sigma_nsq) / (Sl + sigma_nsq)) * (1.0 + oversharp), clamp(Sh / Sl, 0.0, 1.0), select(0.0, 1.0, Sl > Sh));
    return vec4<f32>(avg[0], R);
}

@fragment
fn fs_down_final(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let taps = 3.0;

    let low_x = ceil(-0.5 * taps - 0.5);
    let high_x = floor(0.5 * taps - 0.5);
    let low_y = ceil(-0.5 * taps - 0.5);
    let high_y = floor(0.5 * taps - 0.5);

    let zero3 = vec3<f32>(0.0);
    var avg = mat3x3<f32>(zero3, zero3, zero3);
    var W = 0.0;
    var pos = uv;

    var j: f32 = low_y;
    while (j <= high_y) {
        pos[1] = uv[1] + params.canvasTexelSize[1] * j;
        let rel_y = j;
        let wy = KernelExp(abs(rel_y));

        var row = mat3x3<f32>(zero3, zero3, zero3);
        var rowW = 0.0;
        var pos2 = pos;

        var i: f32 = low_x;
        while (i <= high_x) {
            pos2[0] = uv[0] + params.canvasTexelSize[0] * i;
            let rel_x = i;
            let wx = KernelExp(abs(rel_x));

            let MR = textureSampleLevel(texB, inputSampler, pos2, 0.0);
            row = row + wx * mat3x3<f32>(MR.a * MR.rgb, MR.rgb, vec3<f32>(MR.a));
            rowW = rowW + wx;
            i = i + 1.0;
        }
        row = mat3x3<f32>(row[0] / rowW, row[1] / rowW, row[2] / rowW);
        avg = avg + wy * row;
        W = W + wy;
        j = j + 1.0;
    }
    avg = mat3x3<f32>(avg[0] / W, avg[1] / W, avg[2] / W);

    let L = textureSampleLevel(texA, inputSampler, uv, 0.0);
    return vec4<f32>(avg[1] + avg[2] * L.rgb - avg[0], L.a);
}

@fragment
fn fs_sr_lowres_v(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let axis: i32 = 1;
    let taps = 2.0;

    let base = uv;
    let low_v = ceil((uv - vec2<f32>(taps) / params.videoSize) * params.canvasSize - vec2<f32>(0.5));
    let high_v = floor((uv + vec2<f32>(taps) / params.videoSize) * params.canvasSize - vec2<f32>(0.5));

    var W = 0.0;
    var avg = vec4<f32>(0.0);
    var pos = base;

    let lo = low_v[axis];
    let hi = high_v[axis];

    var k: f32 = lo;
    while (k <= hi) {
        pos[axis] = params.canvasTexelSize[axis] * (k + 0.5);
        let rel = (pos[axis] - base[axis]) * params.videoSize[axis];
        let w = KernelSR(rel);
        var tex = textureSampleLevel(texA, inputSampler, pos, 0.0);
        tex.a = LumaSq(tex.rgb);
        avg = avg + w * tex;
        W = W + w;
        k = k + 1.0;
    }
    avg = avg / W;
    return vec4<f32>(avg.rgb, max(abs(avg.a - LumaSq(avg.rgb)), 5e-7));
}

@fragment
fn fs_sr_lowres_h(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let axis: i32 = 0;
    let taps = 2.0;

    let base = uv;
    let low_v = ceil((uv - vec2<f32>(taps) / params.videoSize) * params.inputTexSize - vec2<f32>(0.5));
    let high_v = floor((uv + vec2<f32>(taps) / params.videoSize) * params.inputTexSize - vec2<f32>(0.5));

    var W = 0.0;
    var avg = vec4<f32>(0.0);
    var pos = base;

    let lo = low_v[axis];
    let hi = high_v[axis];

    var k: f32 = lo;
    while (k <= hi) {
        pos[axis] = params.inputTexTexelSize[axis] * (k + 0.5);
        let rel = (pos[axis] - base[axis]) * params.videoSize[axis];
        let w = KernelSR(rel);
        var tex = textureSampleLevel(texA, inputSampler, pos, 0.0);
        tex.a = LumaSq(tex.rgb);
        avg = avg + w * tex;
        W = W + w;
        k = k + 1.0;
    }
    avg = avg / W;
    let oldA = textureSampleLevel(texA, inputSampler, uv, 0.0).a;
    return vec4<f32>(avg.rgb, max(abs(avg.a - LumaSq(avg.rgb)), 5e-7) + oldA);
}

@fragment
fn fs_sr_var(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let spread = 1.0 / 4.0;
    let vts = params.videoTexelSize;

    let mL0 = textureSampleLevel(texA, inputSampler, uv, 0.0).rgb;
    let mH0 = textureSampleLevel(texB, inputSampler, uv, 0.0);
    var meanL = mL0;
    var meanH = mH0.rgb;

    meanL = meanL + textureSampleLevel(texA, inputSampler, uv + vec2<f32>( 1.0, 0.0) * vts, 0.0).rgb * spread;
    meanL = meanL + textureSampleLevel(texA, inputSampler, uv + vec2<f32>(-1.0, 0.0) * vts, 0.0).rgb * spread;
    meanL = meanL + textureSampleLevel(texA, inputSampler, uv + vec2<f32>( 0.0,  1.0) * vts, 0.0).rgb * spread;
    meanL = meanL + textureSampleLevel(texA, inputSampler, uv + vec2<f32>( 0.0, -1.0) * vts, 0.0).rgb * spread;
    meanH = meanH + textureSampleLevel(texB, inputSampler, uv + vec2<f32>( 1.0, 0.0) * vts, 0.0).rgb * spread;
    meanH = meanH + textureSampleLevel(texB, inputSampler, uv + vec2<f32>(-1.0, 0.0) * vts, 0.0).rgb * spread;
    meanH = meanH + textureSampleLevel(texB, inputSampler, uv + vec2<f32>( 0.0,  1.0) * vts, 0.0).rgb * spread;
    meanH = meanH + textureSampleLevel(texB, inputSampler, uv + vec2<f32>( 0.0, -1.0) * vts, 0.0).rgb * spread;

    meanL = meanL / (1.0 + 4.0 * spread);
    meanH = meanH / (1.0 + 4.0 * spread);

    var varv = vec2<f32>(0.0);
    let mL = textureSampleLevel(texA, inputSampler, uv, 0.0).rgb;
    let mH = textureSampleLevel(texB, inputSampler, uv, 0.0).rgb;
    varv = varv + vec2<f32>(LumaSq(mL - meanL), LumaSq(mH - meanH));

    varv = varv + vec2<f32>(
        LumaSq(textureSampleLevel(texA, inputSampler, uv + vec2<f32>( 1.0, 0.0) * vts, 0.0).rgb - meanL),
        LumaSq(textureSampleLevel(texB, inputSampler, uv + vec2<f32>( 1.0, 0.0) * vts, 0.0).rgb - meanH)
    ) * spread;
    varv = varv + vec2<f32>(
        LumaSq(textureSampleLevel(texA, inputSampler, uv + vec2<f32>(-1.0, 0.0) * vts, 0.0).rgb - meanL),
        LumaSq(textureSampleLevel(texB, inputSampler, uv + vec2<f32>(-1.0, 0.0) * vts, 0.0).rgb - meanH)
    ) * spread;
    varv = varv + vec2<f32>(
        LumaSq(textureSampleLevel(texA, inputSampler, uv + vec2<f32>( 0.0,  1.0) * vts, 0.0).rgb - meanL),
        LumaSq(textureSampleLevel(texB, inputSampler, uv + vec2<f32>( 0.0,  1.0) * vts, 0.0).rgb - meanH)
    ) * spread;
    varv = varv + vec2<f32>(
        LumaSq(textureSampleLevel(texA, inputSampler, uv + vec2<f32>( 0.0, -1.0) * vts, 0.0).rgb - meanL),
        LumaSq(textureSampleLevel(texB, inputSampler, uv + vec2<f32>( 0.0, -1.0) * vts, 0.0).rgb - meanH)
    ) * spread;

    varv = varv / (1.0 + 4.0 * spread);
    varv = max(varv, vec2<f32>(1e-6));
    return vec4<f32>(varv, 0.0, 0.0);
}

@fragment
fn fs_sr_final(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let oversharp = 0.5;
    let taps = 3.0;
    let minX: i32 = -1;
    let maxX: i32 = 1;

    var c0 = textureSampleLevel(texA, inputSampler, uv, 0.0);

    var ipos = uv * params.videoSize - vec2<f32>(0.5);
    let even = (taps - 2.0 * floor(taps / 2.0) == 0.0);
    let offset = ipos - select(floor(ipos + vec2<f32>(0.5)), floor(ipos), even);
    ipos = ipos - offset;

    let halfV = vec2<f32>(0.5);

    var mVar = vec2<f32>(0.0);
    {
        var Y: i32 = -1;
        while (Y <= 1) {
            var X: i32 = -1;
            while (X <= 1) {
                let d = vec2<f32>(f32(X), f32(Y));
                let w = clamp(vec2<f32>(1.5) - abs(d), vec2<f32>(0.0), vec2<f32>(1.0));
                let samplePos = (ipos + d + halfV) * params.videoTexelSize;
                let H = textureSampleLevel(texC, inputSampler, samplePos, 0.0);
                mVar = mVar + w.x * w.y * vec2<f32>(H.a, 1.0);
                X = X + 1;
            }
            Y = Y + 1;
        }
    }
    mVar.x = mVar.x / mVar.y;

    var weightSum = 0.0;
    var diff = vec3<f32>(0.0);

    var Y: i32 = minX;
    while (Y <= maxX) {
        var X: i32 = minX;
        while (X <= maxX) {
            let d = vec2<f32>(f32(X), f32(Y));
            let samplePos = (ipos + d + halfV) * params.videoTexelSize;
            let v = textureSampleLevel(texD, inputSampler, samplePos, 0.0).rg;
            let R = (-1.0 - oversharp) * sqrt(v.x / (v.y + mVar.x));

            let krnl = KernelHann(d.x - offset.x) * KernelHann(d.y - offset.y);
            let H = textureSampleLevel(texC, inputSampler, samplePos, 0.0);
            let Lrgb = textureSampleLevel(texB, inputSampler, samplePos, 0.0).rgb;

            let weight = krnl / (LumaSq(c0.rgb - H.rgb) + H.a);
            diff = diff + weight * (Lrgb + H.rgb * R + (-1.0 - R) * c0.rgb);
            weightSum = weightSum + weight;
            X = X + 1;
        }
        Y = Y + 1;
    }
    diff = diff / weightSum;

    c0 = vec4<f32>(c0.rgb + diff, c0.a);
    return c0;
}
    `;

    const shaderModule = device.createShaderModule({ code: wgslCode });
    console.log("[SSim] WGSL shader compiled");

    const linearSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
    });

    const UNIFORM_SIZE = 64;
    const UNIFORM_ALIGN = 256;
    const NUM_SLOTS = 9;
    const uniformBuffer = device.createBuffer({
        size: UNIFORM_ALIGN * NUM_SLOTS,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    function createTex(w, h, format) {
        return device.createTexture({
            size: [Math.max(1, w), Math.max(1, h), 1],
            format: format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    function makePipeline(entryPoint, format) {
        return device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: { module: shaderModule, entryPoint: entryPoint, targets: [{ format: format }] },
            primitive: { topology: 'triangle-list' },
        });
    }

    function makeBindGroup(pipeline, entries) {
        return device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: entries,
        });
    }

    async function hijackVideo(video) {
        if (video.dataset.ssimHijacked) return;
        if (!video.currentSrc && !video.src) {
            video.addEventListener('loadedmetadata', () => hijackVideo(video), { once: true });
            return;
        }
        cleanupActive(true);
        const seq = ++hijackSeq;
        const src = video.currentSrc || video.src || "no src";
        console.log("[SSim] Video found, src:", src.substring(0, 80));
        video.dataset.ssimHijacked = "true";
        video.crossOrigin = "anonymous";

        const canvas = document.createElement('canvas');
        canvas.style.position = 'absolute';
        canvas.style.pointerEvents = 'none';
        canvas.style.transition = 'opacity 0s';
        const videoZIndex = getComputedStyle(video).zIndex;
        canvas.style.zIndex = videoZIndex;
        video.parentNode.insertBefore(canvas, video.nextSibling);
        video.style.opacity = '0';

        activeState = { video, canvas, running: true, seq };

        console.log("[SSim] Building SSim WebGPU render layer...");
        const context = canvas.getContext('webgpu');
        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format: presentationFormat, alphaMode: 'premultiplied' });

        const pBilinear = makePipeline('fs_bilinear', RGBA16F);
        const pDownL2V = makePipeline('fs_down_l2_v', RGBA16F);
        const pDownL2H = makePipeline('fs_down_l2_h', RGBA16F);
        const pDownMR = makePipeline('fs_down_mr', RGBA16F);
        const pDownFinal = makePipeline('fs_down_final', presentationFormat);
        const pSRLowresV = makePipeline('fs_sr_lowres_v', RGBA16F);
        const pSRLowresH = makePipeline('fs_sr_lowres_h', RGBA16F);
        const pSRVar = makePipeline('fs_sr_var', RGBA16F);
        const pSRFinal = makePipeline('fs_sr_final', presentationFormat);
        console.log("[SSim] All pipelines created, format:", presentationFormat);

        let frameCount = 0;
        let lastVideoRes = "";
        let lastCanvasRes = "";
        let lastMode = "";
        const mySeq = seq;

        let videoTexture = null;
        let texState = { key: "" };
        let bindGroups = {};

        function ensureTextures(vw, vh, cw, ch) {
            const key = vw + "_" + vh + "_" + cw + "_" + ch;
            if (texState.key === key) return;
            texState.key = key;

            if (videoTexture) videoTexture.destroy();
            videoTexture = device.createTexture({
                size: [vw, vh, 1],
                format: RGBA16F,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
            });

            if (texState.postKernel) texState.postKernel.destroy();
            if (texState.l2A) texState.l2A.destroy();
            if (texState.l2B) texState.l2B.destroy();
            if (texState.mr) texState.mr.destroy();
            if (texState.lowresA) texState.lowresA.destroy();
            if (texState.lowresB) texState.lowresB.destroy();
            if (texState.var) texState.var.destroy();

            texState.postKernel = createTex(cw, ch, RGBA16F);
            texState.l2A = createTex(vw, ch, RGBA16F);
            texState.l2B = createTex(cw, ch, RGBA16F);
            texState.mr = createTex(cw, ch, RGBA16F);
            texState.lowresA = createTex(cw, vh, RGBA16F);
            texState.lowresB = createTex(vw, vh, RGBA16F);
            texState.var = createTex(vw, vh, RGBA16F);

            bindGroups = {};

            console.log("[SSim] Textures (re)created for", vw + "x" + vh, "->", cw + "x" + ch);
        }

        function getBG(name, pipeline, slot, resources, hasUniform) {
            if (bindGroups[name]) return bindGroups[name];
            const entries = [];
            if (hasUniform !== false) {
                entries.push({ binding: 0, resource: { buffer: uniformBuffer, offset: slot * UNIFORM_ALIGN, size: UNIFORM_SIZE } });
            }
            entries.push({ binding: 1, resource: linearSampler });
            for (let i = 0; i < resources.length; i++) {
                entries.push({ binding: 2 + i, resource: resources[i] });
            }
            bindGroups[name] = makeBindGroup(pipeline, entries);
            return bindGroups[name];
        }

        function writeUniforms(slot, videoW, videoH, canvasW, canvasH, inputTexW, inputTexH) {
            const data = new Float32Array(16);
            data[0] = videoW; data[1] = videoH;
            data[2] = 1.0 / videoW; data[3] = 1.0 / videoH;
            data[4] = canvasW; data[5] = canvasH;
            data[6] = 1.0 / canvasW; data[7] = 1.0 / canvasH;
            data[8] = inputTexW; data[9] = inputTexH;
            data[10] = 1.0 / inputTexW; data[11] = 1.0 / inputTexH;
            device.queue.writeBuffer(uniformBuffer, slot * UNIFORM_ALIGN, data);
        }

        function runPass(encoder, pipeline, bindGroup, targetView, clearAlpha) {
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: targetView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: clearAlpha ? 1.0 : 0.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.draw(6);
            pass.end();
        }

        function frame() {
            if (hasWebGPUError) {
                if (Date.now() - lastErrorTime > 5000) {
                    hasWebGPUError = false;
                } else {
                    cleanupActive(true);
                    return;
                }
            }
            if (!activeState || !activeState.running || activeState.seq !== mySeq) {
                video.requestVideoFrameCallback(frame);
                return;
            }
            const v = activeState.video;
            const c = activeState.canvas;

            if (!v.isConnected) { cleanupActive(true); return; }

            if (v.paused || v.ended || v.readyState < 2 || v.videoWidth === 0) {
                video.requestVideoFrameCallback(frame);
                return;
            }

            const resKey = v.videoWidth + "x" + v.videoHeight;
            if (resKey !== lastVideoRes) {
                console.log("[SSim] Source resolution:", resKey);
                lastVideoRes = resKey;
            }

            const dpr = window.devicePixelRatio || 1;
            const targetWidth = Math.round(v.offsetWidth * dpr);
            const targetHeight = Math.round(v.offsetHeight * dpr);

            const srcW = v.videoWidth;
            const srcH = v.videoHeight;
            const dstW = targetWidth;
            const dstH = targetHeight;

            const isDownscaling = dstW < srcW || dstH < srcH;
            const isUpscaling = dstW > srcW || dstH > srcH;

            if (!isDownscaling && !isUpscaling) {
                v.style.opacity = '1';
                c.style.opacity = '0';
                video.requestVideoFrameCallback(frame);
                return;
            }

            const mode = isDownscaling ? "downscale" : "superres";
            if (mode !== lastMode) {
                console.log("[SSim] Mode:", mode);
                lastMode = mode;
            }

            const canvasRes = dstW + "x" + dstH;
            if (canvasRes !== lastCanvasRes) {
                console.log("[SSim] Output resolution:", canvasRes, "@", dpr + "x DPR");
                lastCanvasRes = canvasRes;
            }
            if (c.width !== dstW || c.height !== dstH) {
                c.width = dstW;
                c.height = dstH;
                context.configure({ device, format: presentationFormat, alphaMode: 'premultiplied' });
            }

            const vr = v.getBoundingClientRect();
            let cr = c.parentNode;
            while (cr && cr !== document.documentElement && cr !== document.body) {
                const p = getComputedStyle(cr).position;
                if (p !== 'static') break;
                cr = cr.parentNode;
            }
            const cb = cr.getBoundingClientRect ? cr.getBoundingClientRect() : { left: 0, top: 0 };
            const ar = srcW / srcH;
            const cssAr = vr.width / vr.height;
            let w = vr.width;
            let h = vr.height;
            let left = vr.left - cb.left;
            let top = vr.top - cb.top;
            if (Math.abs(cssAr - ar) > 0.001) {
                if (cssAr > ar) {
                    w = Math.round(vr.height * ar);
                    left += (vr.width - w) / 2;
                } else {
                    h = Math.round(vr.width / ar);
                    top += (vr.height - h) / 2;
                }
            }
            c.style.width = w + 'px';
            c.style.height = h + 'px';
            c.style.left = left + 'px';
            c.style.top = top + 'px';

            v.style.opacity = '0';
            c.style.opacity = '1';

            try {
                ensureTextures(srcW, srcH, dstW, dstH);
                device.queue.copyExternalImageToTexture({ source: v }, { texture: videoTexture }, [srcW, srcH]);
            } catch (e) {
                v.requestVideoFrameCallback(frame);
                return;
            }

            const encoder = device.createCommandEncoder();

            if (isDownscaling) {
                writeUniforms(0, srcW, srcH, dstW, dstH, srcW, srcH);
                runPass(encoder, pBilinear,
                    getBG('bilinear', pBilinear, 0, [videoTexture.createView()], false),
                    texState.postKernel.createView(), false);

                writeUniforms(1, srcW, srcH, dstW, dstH, srcW, srcH);
                runPass(encoder, pDownL2V,
                    getBG('down_l2_v', pDownL2V, 1, [videoTexture.createView()]),
                    texState.l2A.createView(), false);

                writeUniforms(2, srcW, srcH, dstW, dstH, srcW, dstH);
                runPass(encoder, pDownL2H,
                    getBG('down_l2_h', pDownL2H, 2, [texState.l2A.createView()]),
                    texState.l2B.createView(), false);

                writeUniforms(3, srcW, srcH, dstW, dstH, dstW, dstH);
                runPass(encoder, pDownMR,
                    getBG('down_mr', pDownMR, 3, [texState.postKernel.createView(), texState.l2B.createView()]),
                    texState.mr.createView(), false);

                writeUniforms(4, srcW, srcH, dstW, dstH, dstW, dstH);
                runPass(encoder, pDownFinal,
                    getBG('down_final', pDownFinal, 4, [texState.postKernel.createView(), texState.mr.createView()]),
                    context.getCurrentTexture().createView(), true);
            } else {
                writeUniforms(0, srcW, srcH, dstW, dstH, srcW, srcH);
                runPass(encoder, pBilinear,
                    getBG('bilinear', pBilinear, 0, [videoTexture.createView()], false),
                    texState.postKernel.createView(), false);

                writeUniforms(5, srcW, srcH, dstW, dstH, dstW, dstH);
                runPass(encoder, pSRLowresV,
                    getBG('sr_lowres_v', pSRLowresV, 5, [texState.postKernel.createView()]),
                    texState.lowresA.createView(), false);

                writeUniforms(6, srcW, srcH, dstW, dstH, dstW, srcH);
                runPass(encoder, pSRLowresH,
                    getBG('sr_lowres_h', pSRLowresH, 6, [texState.lowresA.createView()]),
                    texState.lowresB.createView(), false);

                writeUniforms(7, srcW, srcH, dstW, dstH, srcW, srcH);
                runPass(encoder, pSRVar,
                    getBG('sr_var', pSRVar, 7, [videoTexture.createView(), texState.lowresB.createView()]),
                    texState.var.createView(), false);

                writeUniforms(8, srcW, srcH, dstW, dstH, dstW, dstH);
                runPass(encoder, pSRFinal,
                    getBG('sr_final', pSRFinal, 8, [
                        texState.postKernel.createView(),
                        videoTexture.createView(),
                        texState.lowresB.createView(),
                        texState.var.createView(),
                    ]),
                    context.getCurrentTexture().createView(), true);
            }

            device.queue.submit([encoder.finish()]);

            if (frameCount++ === 0) {
                console.log("[SSim] First frame rendered successfully");
            }

            v.requestVideoFrameCallback(frame);
        }

        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            video.requestVideoFrameCallback(frame);
        }
    }

    if (document.readyState === 'loading') {
        console.log("[SSim] Waiting for DOMContentLoaded...");
        await new Promise(resolve => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
    }
    console.log("[SSim] Page ready, starting observer");
    const observer = new MutationObserver(() => document.querySelectorAll('video').forEach(hijackVideo));
    observer.observe(document.body, { childList: true, subtree: true });
    document.querySelectorAll('video').forEach(hijackVideo);

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && activeState) {
            activeState.running = false;
        } else if (!document.hidden) {
            if (activeState) {
                activeState.running = true;
            } else {
                document.querySelectorAll('video').forEach(v => {
                    if (v.dataset.ssimHijacked) {
                        delete v.dataset.ssimHijacked;
                    }
                });
                document.querySelectorAll('video').forEach(hijackVideo);
            }
        }
    });

    document.addEventListener('fullscreenchange', () => {
        if (!activeState || !activeState.canvas) return;
        const fs = document.fullscreenElement;
        const c = activeState.canvas;
        if (fs && fs !== document.body && !fs.contains(c)) {
            fs.appendChild(c);
        } else if (!fs) {
            const v = activeState.video;
            if (v && v.isConnected && v.parentNode) {
                v.parentNode.insertBefore(c, v.nextSibling);
            } else {
                document.body.appendChild(c);
            }
        }
    });
})();
