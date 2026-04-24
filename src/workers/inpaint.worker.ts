/**
 * inpaint.worker.ts — Hybrid inpainting
 *
 * Two strategies, chosen per bubble via isBrightRegion():
 *
 *   bright interior → speech bubble  → paint tight text rect white
 *     Fast, no model needed. Writes white pixels to the output overlay.
 *
 *   dark/colored    → background text → LaMa ONNX
 *     Text floats on manga artwork.
 *     LaMa reconstructs the background behind the text.
 *     Writes reconstructed pixels to the output overlay.
 *
 * Output: transparent PNG overlay (same dims as input).
 *   - Speech bubble text rects  → white (255,255,255,255)
 *   - Background text regions   → LaMa-reconstructed pixels
 *   - Everything else           → transparent (0,0,0,0)
 * The UI stamps this directly onto the .ws-inpaint-layer canvas.
 *
 * Model: dreMaz/AnimeMangaInpainting / lama_manga_fp32.onnx (~199 MB, OPFS-cached after first use)
 *         LaMa fine-tuned on 300k manga+anime images. Input/output: 0–255 float32.
 *
 * Message protocol:
 *   IN  { type: 'inpaint', imageBlob: Blob, bubbles: Array<{id, rect: {x,y,w,h}}> }
 *         rect values are percentage-based (0–100) matching MangaBubble.rect
 *   OUT { type: 'progress', current: number, total: number, stage: string }
 *   OUT { type: 'done',     resultBlob: Blob, expandedRects: Array<{id, rect: {x,y,w,h}}> }
 *         expandedRects only contains entries for speech bubble route (bright interior).
 *         rect values are percentage-based, covering the full bubble interior.
 *   OUT { type: 'error',    message: string }
 */

import * as ort from 'onnxruntime-web'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/'

// Set to true to re-enable LaMa background reconstruction (requires 199 MB model download).
const LAMA_ENABLED  = false

const MODEL_URL     = '/lama_manga_fp32.onnx'
const OPFS_FILENAME = 'lama_manga_fp32.onnx'
const LAMA_SIZE       = 512
const LAMA_CTX_FRAC   = 0.5   // context padding around background text for LaMa (fraction of max(w,h))
const BG_PADDING      = 8     // px padding around tight text rect for LaMa bounds

const TEXT_LUM_THRESH  = 160  // pixels brighter than this inside the bounds are treated as text ink
const BRIGHT_THRESH    = 200  // luminance above this = bright pixel
const BRIGHT_MIN_RATIO = 0.50 // fraction of interior samples that must be bright to confirm speech bubble

const SOLID_RING       = 12  // px wide sampling ring around text rect for solid-bg detection
const SOLID_RING_SKIP  = 5   // skip this many px right outside the text rect (white character halos live here)
const SOLID_THRESH     = 120 // max per-channel stddev — below this = solid/uniform background (120 catches screentone halftone patterns)
const HALO_WHITE_THRESH = 235 // pixels brighter than this in the sample ring are treated as halo contamination and excluded

// ── Bubble boundary scanner ────────────────────────────────────────────────────
// Used for speech bubbles only: expands the tight text rect outward until
// hitting the dark bubble border, returning the full bubble interior bounds.

const DARK_THRESH  = 80
const MAX_EXPAND   = 200
const SCAN_SAMPLES = 9
const WHITE_EXPAND = 4   // px to expand tight text rect when painting white (clamped to bubble bounds)

function scanBubbleBounds(
  pixels: Uint8ClampedArray,
  W: number, H: number,
  x1: number, y1: number, x2: number, y2: number,
): [number, number, number, number] {
  function lum(x: number, y: number): number {
    const xi = Math.max(0, Math.min(W - 1, Math.round(x)))
    const yi = Math.max(0, Math.min(H - 1, Math.round(y)))
    const b  = (yi * W + xi) * 4
    return pixels[b] * 0.299 + pixels[b + 1] * 0.587 + pixels[b + 2] * 0.114
  }
  function scanEdge(
    edge: number, step: number, limit: number,
    perpStart: number, perpEnd: number, horizontal: boolean,
  ): number {
    let minDist = MAX_EXPAND
    for (let s = 0; s < SCAN_SAMPLES; s++) {
      const perp = perpStart + (perpEnd - perpStart) * s / (SCAN_SAMPLES - 1)
      let dist = 0
      for (let d = 1; d <= MAX_EXPAND; d++) {
        const coord = edge + step * d
        if (coord < 0 || coord >= limit) { dist = d; break }
        const brightness = horizontal ? lum(coord, perp) : lum(perp, coord)
        if (brightness < DARK_THRESH) { dist = d - 1; break }
        dist = d
      }
      if (dist < minDist) minDist = dist
    }
    return minDist
  }
  const leftExp  = scanEdge(x1, -1, W, y1, y2, true)
  const rightExp = scanEdge(x2,  1, W, y1, y2, true)
  const topExp   = scanEdge(y1, -1, H, x1, x2, false)
  const botExp   = scanEdge(y2,  1, H, x1, x2, false)
  return [
    Math.max(0, x1 - leftExp),      Math.max(0, y1 - topExp),
    Math.min(W - 1, x2 + rightExp), Math.min(H - 1, y2 + botExp),
  ]
}

// ── Background brightness check ───────────────────────────────────────────────
// Samples a 5×5 grid INSIDE the tight text rect.
// Speech bubbles have white backgrounds between strokes → high bright-ratio.
// Background text sits on dark/colored artwork → low bright-ratio.
// Sampling interior avoids being fooled by dark panel borders outside the bubble.

function isBrightRegion(
  pixels: Uint8ClampedArray,
  W: number, H: number,
  tx1: number, ty1: number, tx2: number, ty2: number,
): boolean {
  const GRID = 5
  let bright = 0
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const sx = Math.max(0, Math.min(W - 1, Math.round(tx1 + (tx2 - tx1) * (gx + 0.5) / GRID)))
      const sy = Math.max(0, Math.min(H - 1, Math.round(ty1 + (ty2 - ty1) * (gy + 0.5) / GRID)))
      const b  = (sy * W + sx) * 4
      const l  = pixels[b] * 0.299 + pixels[b + 1] * 0.587 + pixels[b + 2] * 0.114
      if (l > BRIGHT_THRESH) bright++
    }
  }
  return (bright / (GRID * GRID)) >= BRIGHT_MIN_RATIO
}

// ── Background color from heatmap ─────────────────────────────────────────────
// When the detection heatmap is available, sample only the NON-text pixels
// (mask=0) in the region + padding. These are guaranteed to be background.
// Returns mean RGB, mean luminance, and whether the background is solid (low variance).

function sampleBackgroundFromMask(
  pixels: Uint8ClampedArray,
  W: number, H: number,
  tx1: number, ty1: number, tx2: number, ty2: number,
  textMask: Uint8Array,
): { r: number; g: number; b: number; lum: number; solid: boolean; dilation: number; rough_bg_lum: number; halo_dilation: number } {
  const mean   = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length
  const stddev = (a: number[], m: number) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length)
  const STEP = 3

  // Adaptive halo dilation: check the 3–6px annulus around text pixels.
  // If bgLum is dark (<180) and that annulus is mainly bright (>50% pixels lum>200),
  // the artist drew thick white outlines → use 6px dilation.
  // Otherwise 3px is enough (no halo or thin halo or bright background).
  const ix1 = Math.max(0, tx1), iy1 = Math.max(0, ty1)
  const ix2 = Math.min(W - 1, tx2), iy2 = Math.min(H - 1, ty2)

  // Quick rough bgLum estimate from inner non-text pixels at coarse step
  let roughLumSum = 0, roughLumCount = 0
  for (let y = iy1; y <= iy2; y += 6) {
    for (let x = ix1; x <= ix2; x += 6) {
      if (textMask[y * W + x] > 0) continue
      const p = (y * W + x) * 4
      roughLumSum += pixels[p] * 0.299 + pixels[p + 1] * 0.587 + pixels[p + 2] * 0.114
      roughLumCount++
    }
  }
  const roughBgLum = roughLumCount > 0 ? roughLumSum / roughLumCount : 128

  let HALO_DILATION = 3
  if (roughBgLum < 180) {
    // Check the 3–6px annulus: pixels near text but beyond the base dilation
    let annulusBright = 0, annulusTotal = 0
    for (let y = iy1; y <= iy2; y += STEP) {
      for (let x = ix1; x <= ix2; x += STEP) {
        let inOuter = false, inInner = false
        for (let dy = -6; dy <= 6 && !inOuter; dy++)
          for (let dx = -6; dx <= 6 && !inOuter; dx++) {
            const ny = y + dy, nx = x + dx
            if (ny >= 0 && ny < H && nx >= 0 && nx < W && textMask[ny * W + nx] > 0) {
              const dist = Math.max(Math.abs(dy), Math.abs(dx))
              if (dist <= 6) inOuter = true
              if (dist <= 3) inInner = true
            }
          }
        if (!inOuter || inInner) continue  // only pixels in the 3–6px annulus
        const p = (y * W + x) * 4
        const lum = pixels[p] * 0.299 + pixels[p + 1] * 0.587 + pixels[p + 2] * 0.114
        annulusTotal++
        if (lum > 200) annulusBright++
      }
    }
    if (annulusTotal > 0 && annulusBright / annulusTotal > 0.5) HALO_DILATION = 6
  }

  // Sample only the narrow band just outside the halo zone — pixels whose
  // Chebyshev distance to the nearest text stroke is in [HALO_DILATION+1, HALO_DILATION+SAMPLE_BAND].
  // This is the immediate background neighboring the text, most representative for routing + color.
  const SAMPLE_BAND = 8
  const OUTER = HALO_DILATION + SAMPLE_BAND
  const sx1 = Math.max(0, tx1 - OUTER), sy1 = Math.max(0, ty1 - OUTER)
  const sx2 = Math.min(W - 1, tx2 + OUTER), sy2 = Math.min(H - 1, ty2 + OUTER)
  const rs: number[] = [], gs: number[] = [], bs: number[] = []
  for (let y = sy1; y <= sy2; y += STEP) {
    for (let x = sx1; x <= sx2; x += STEP) {
      let inHalo = false, inBand = false
      for (let dy = -OUTER; dy <= OUTER; dy++) {
        for (let dx = -OUTER; dx <= OUTER; dx++) {
          const ny = y + dy, nx = x + dx
          if (ny < 0 || ny >= H || nx < 0 || nx >= W || textMask[ny * W + nx] === 0) continue
          const dist = Math.max(Math.abs(dy), Math.abs(dx))
          if (dist <= HALO_DILATION) { inHalo = true; break }
          if (dist <= OUTER) inBand = true
        }
        if (inHalo) break
      }
      if (inHalo || !inBand) continue
      const p = (y * W + x) * 4
      rs.push(pixels[p]); gs.push(pixels[p + 1]); bs.push(pixels[p + 2])
    }
  }

  // Fallback: if too few inner pixels (dense text), sample the outer ring instead
  if (rs.length < 4) {
    const bx1 = Math.max(0, tx1 - SOLID_RING), by1 = Math.max(0, ty1 - SOLID_RING)
    const bx2 = Math.min(W - 1, tx2 + SOLID_RING), by2 = Math.min(H - 1, ty2 + SOLID_RING)
    for (let y = by1; y <= by2; y += STEP) {
      for (let x = bx1; x <= bx2; x += STEP) {
        if (textMask[y * W + x] > 0) continue
        const p = (y * W + x) * 4
        rs.push(pixels[p]); gs.push(pixels[p + 1]); bs.push(pixels[p + 2])
      }
    }
  }

  if (rs.length < 4) return { r: 128, g: 128, b: 128, lum: 128, solid: false, dilation: HALO_DILATION, rough_bg_lum: roughBgLum, halo_dilation: HALO_DILATION }

  // Mode (binned by 8) for fill color — finds the dominant background value
  // robustly, ignoring any remaining outliers.
  const modeBin = (a: number[]) => {
    const counts = new Map<number, number>()
    for (const v of a) { const bin = Math.round(v / 8) * 8; counts.set(bin, (counts.get(bin) ?? 0) + 1) }
    let best = a[0], bestCount = 0
    for (const [bin, count] of counts) { if (count > bestCount) { bestCount = count; best = bin } }
    return Math.min(255, best)
  }
  const mr = modeBin(rs), mg = modeBin(gs), mb = modeBin(bs)
  // stddev uses mean — measures spread for solid detection
  const meanR = mean(rs), meanG = mean(gs), meanB = mean(bs)
  const lum  = mr * 0.299 + mg * 0.587 + mb * 0.114
  const solid = Math.max(stddev(rs, meanR), stddev(gs, meanG), stddev(bs, meanB)) < SOLID_THRESH
  return { r: mr, g: mg, b: mb, lum, solid, dilation: HALO_DILATION, rough_bg_lum: roughBgLum, halo_dilation: HALO_DILATION }
}

// ── Solid background detection ────────────────────────────────────────────────
// Samples a ring of pixels just outside the tight text rect.
// If all channels have low stddev the background is uniform → fill with average color.
// Returns { r, g, b, solid } — solid=false means complex artwork → fall through to LaMa.

function sampleBorderColor(
  pixels: Uint8ClampedArray,
  W: number, H: number,
  tx1: number, ty1: number, tx2: number, ty2: number,
): { r: number; g: number; b: number; solid: boolean } {
  // Outer ring boundary (up to SOLID_RING px from text rect edge)
  const bx1 = Math.max(0, tx1 - SOLID_RING)
  const by1 = Math.max(0, ty1 - SOLID_RING)
  const bx2 = Math.min(W - 1, tx2 + SOLID_RING)
  const by2 = Math.min(H - 1, ty2 + SOLID_RING)
  // Inner exclusion zone: text rect + SOLID_RING_SKIP px (skips character white halos)
  const ix1 = tx1 - SOLID_RING_SKIP, iy1 = ty1 - SOLID_RING_SKIP
  const ix2 = tx2 + SOLID_RING_SKIP, iy2 = ty2 + SOLID_RING_SKIP
  const STEP = 4
  const rs: number[] = [], gs: number[] = [], bs: number[] = []
  for (let y = by1; y <= by2; y += STEP) {
    for (let x = bx1; x <= bx2; x += STEP) {
      if (x >= ix1 && x <= ix2 && y >= iy1 && y <= iy2) continue  // skip halo zone
      const p = (y * W + x) * 4
      const r = pixels[p], g = pixels[p + 1], b = pixels[p + 2]
      // Exclude near-white pixels — likely residual halos or bubble interior bleed
      if (r > HALO_WHITE_THRESH && g > HALO_WHITE_THRESH && b > HALO_WHITE_THRESH) continue
      rs.push(r); gs.push(g); bs.push(b)
    }
  }
  // If no non-white pixels found, the background is white
  if (rs.length === 0) return { r: 255, g: 255, b: 255, solid: true }
  const mean   = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length
  const stddev = (a: number[], m: number) => Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length)
  const mr = mean(rs), mg = mean(gs), mb = mean(bs)
  const solid = Math.max(stddev(rs, mr), stddev(gs, mg), stddev(bs, mb)) < SOLID_THRESH
  return { r: Math.round(mr), g: Math.round(mg), b: Math.round(mb), solid }
}

// ── Pixel helpers ──────────────────────────────────────────────────────────────

function cropPixels(
  src: Uint8ClampedArray, W: number,
  x: number, y: number, w: number, h: number,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4)
  for (let row = 0; row < h; row++) {
    const srcOff = ((y + row) * W + x) * 4
    out.set(src.subarray(srcOff, srcOff + w * 4), row * w * 4)
  }
  return out
}

function scalePixels(
  pixels: Uint8ClampedArray,
  srcW: number, srcH: number,
  dstW: number, dstH: number,
): Uint8ClampedArray {
  const srcCanvas = new OffscreenCanvas(srcW, srcH)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  srcCanvas.getContext('2d')!.putImageData(new ImageData(pixels as any, srcW, srcH), 0, 0)
  const dstCanvas = new OffscreenCanvas(dstW, dstH)
  dstCanvas.getContext('2d')!.drawImage(srcCanvas, 0, 0, dstW, dstH)
  return dstCanvas.getContext('2d')!.getImageData(0, 0, dstW, dstH).data
}

// ── OPFS model cache ───────────────────────────────────────────────────────────

async function loadFromOpfs(): Promise<ArrayBuffer | null> {
  try {
    const root = await navigator.storage.getDirectory()
    const fh = await root.getFileHandle(OPFS_FILENAME)
    return (await fh.getFile()).arrayBuffer()
  } catch {
    return null
  }
}

async function saveToOpfs(buffer: ArrayBuffer): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    const fh = await root.getFileHandle(OPFS_FILENAME, { create: true })
    const w = await fh.createWritable()
    await w.write(buffer)
    await w.close()
  } catch { /* OPFS unavailable — skip cache */ }
}

// ── Singleton ONNX session ─────────────────────────────────────────────────────

let session: ort.InferenceSession | null = null

async function downloadModel(): Promise<ArrayBuffer> {
  const resp = await fetch(MODEL_URL)
  if (!resp.ok) throw new Error(`Model fetch failed: ${resp.status} ${resp.statusText}`)
  const contentLength = Number(resp.headers.get('content-length') ?? 0)
  const reader = resp.body!.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    if (contentLength > 0) {
      const pct = Math.round(received / contentLength * 100)
      post({ type: 'progress', current: received, total: contentLength,
        stage: `Downloading inpaint model… ${pct}% (199 MB, cached after first run)` })
    }
  }
  const arr = new Uint8Array(received)
  let off = 0
  for (const chunk of chunks) { arr.set(chunk, off); off += chunk.length }
  return arr.buffer
}

async function clearOpfsCache(): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    await root.removeEntry(OPFS_FILENAME)
  } catch { /* already gone */ }
}

async function getSession(): Promise<ort.InferenceSession> {
  if (session) return session

  post({ type: 'progress', current: 0, total: 1, stage: 'Checking LaMa model cache…' })
  let buffer = await loadFromOpfs()

  if (!buffer) {
    buffer = await downloadModel()
    saveToOpfs(buffer)  // fire-and-forget
  }

  post({ type: 'progress', current: 0, total: 1, stage: 'Loading LaMa into ONNX runtime…' })
  try {
    session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] })
  } catch (err) {
    // Cached file may be corrupted — clear it and retry with a fresh download
    await clearOpfsCache()
    post({ type: 'progress', current: 0, total: 1, stage: 'Cache invalid, re-downloading LaMa model…' })
    buffer = await downloadModel()
    saveToOpfs(buffer)
    post({ type: 'progress', current: 0, total: 1, stage: 'Loading LaMa into ONNX runtime…' })
    session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] })
  }
  return session
}

// ── LaMa inference ─────────────────────────────────────────────────────────────

function post(msg: object): void { self.postMessage(msg) }

async function runLama(
  sess: ort.InferenceSession,
  imgPixels: Uint8ClampedArray,  // RGBA, LAMA_SIZE × LAMA_SIZE
  maskPixels: Uint8Array,        // 1-channel, LAMA_SIZE × LAMA_SIZE, 255 = inpaint
): Promise<Float32Array> {
  const N = LAMA_SIZE * LAMA_SIZE
  const imgData  = new Float32Array(3 * N)
  const maskData = new Float32Array(N)
  for (let i = 0; i < N; i++) {
    // Model expects 0–255 input (normalization is done inside the ONNX wrapper)
    imgData[i]         = imgPixels[i * 4]
    imgData[N + i]     = imgPixels[i * 4 + 1]
    imgData[2 * N + i] = imgPixels[i * 4 + 2]
    maskData[i]        = maskPixels[i] > 128 ? 1.0 : 0.0
  }
  const imgName  = sess.inputNames.find(n => /image|img|input/i.test(n))  ?? sess.inputNames[0]
  const maskName = sess.inputNames.find(n => /mask/i.test(n))             ?? sess.inputNames[1]
  const feeds: Record<string, ort.Tensor> = {
    [imgName]:  new ort.Tensor('float32', imgData,  [1, 3, LAMA_SIZE, LAMA_SIZE]),
    [maskName]: new ort.Tensor('float32', maskData, [1, 1, LAMA_SIZE, LAMA_SIZE]),
  }
  const results = await sess.run(feeds)
  return results[sess.outputNames[0]].data as Float32Array
}

// ── Background text path — LaMa ────────────────────────────────────────────────
// bounds is the tight text rect + BG_PADDING from scanBubbleBounds.
// We build a rectangular mask covering bounds, crop a larger context region,
// run LaMa, and paste the reconstructed pixels into outData.

async function inpaintBackground(
  sess: ort.InferenceSession,
  origPixels: Uint8ClampedArray,
  outData: Uint8ClampedArray,
  W: number, H: number,
  bounds: [number, number, number, number],
  textMask?: Uint8Array,
): Promise<void> {
  const [bx, by, bx2, by2] = bounds
  const bw = bx2 - bx
  const bh = by2 - by

  // Context region: expand bounds further so LaMa sees surrounding background
  const pad = Math.round(Math.max(bw, bh) * LAMA_CTX_FRAC)
  const rx1 = Math.max(0, Math.floor(bx) - pad)
  const ry1 = Math.max(0, Math.floor(by) - pad)
  const rx2 = Math.min(W, Math.ceil(bx2) + pad)
  const ry2 = Math.min(H, Math.ceil(by2) + pad)
  const rw  = rx2 - rx1
  const rh  = ry2 - ry1

  // Pixel-level mask: within the bounds, mark text pixels.
  // Prefer the detection heatmap (exact text strokes); fall back to luminance threshold.
  const cropMask = new Uint8Array(rw * rh)
  let maskedCount = 0
  for (let row = 0; row < rh; row++) {
    const gy = ry1 + row
    for (let col = 0; col < rw; col++) {
      const gx = rx1 + col
      if (gx < bx || gx > bx2 || gy < by || gy > by2) continue
      const isText = textMask
        ? textMask[gy * W + gx] > 0
        : (() => {
            const p = (gy * W + gx) * 4
            return origPixels[p] * 0.299 + origPixels[p + 1] * 0.587 + origPixels[p + 2] * 0.114 > TEXT_LUM_THRESH
          })()
      if (isText) { cropMask[row * rw + col] = 255; maskedCount++ }
    }
  }
  // Fallback: if mask found almost nothing, use the full rectangle
  if (maskedCount < (bw * bh) * 0.01) {
    for (let row = 0; row < rh; row++) {
      const gy = ry1 + row
      for (let col = 0; col < rw; col++) {
        const gx = rx1 + col
        if (gx >= bx && gx <= bx2 && gy >= by && gy <= by2) cropMask[row * rw + col] = 255
      }
    }
  }

  // Scale image crop and mask to LAMA_SIZE
  const cropImg   = cropPixels(origPixels, W, rx1, ry1, rw, rh)
  const scaledImg = scalePixels(cropImg, rw, rh, LAMA_SIZE, LAMA_SIZE)

  const maskRgba = new Uint8ClampedArray(rw * rh * 4)
  for (let p = 0; p < rw * rh; p++) {
    maskRgba[p * 4] = maskRgba[p * 4 + 1] = maskRgba[p * 4 + 2] = cropMask[p]
    maskRgba[p * 4 + 3] = 255
  }
  const scaledMaskRgba = scalePixels(maskRgba, rw, rh, LAMA_SIZE, LAMA_SIZE)
  const scaledMask = new Uint8Array(LAMA_SIZE * LAMA_SIZE)
  for (let p = 0; p < LAMA_SIZE * LAMA_SIZE; p++)
    scaledMask[p] = scaledMaskRgba[p * 4] > 128 ? 255 : 0

  // Run LaMa
  const output = await runLama(sess, scaledImg, scaledMask)

  // Convert output float32 [3,512,512] → RGBA, scale back to crop size
  const N = LAMA_SIZE * LAMA_SIZE
  const outRgba = new Uint8ClampedArray(N * 4)
  for (let p = 0; p < N; p++) {
    // Model outputs 0–255 directly (no scaling needed)
    outRgba[p * 4]     = Math.max(0, Math.min(255, Math.round(output[p]        )))
    outRgba[p * 4 + 1] = Math.max(0, Math.min(255, Math.round(output[N + p]    )))
    outRgba[p * 4 + 2] = Math.max(0, Math.min(255, Math.round(output[2 * N + p])))
    outRgba[p * 4 + 3] = 255
  }
  const resultPixels = scalePixels(outRgba, LAMA_SIZE, LAMA_SIZE, rw, rh)

  // Paste the entire bounds region from LaMa into outData.
  // LaMa produces output for all pixels: non-text pixels retain original color,
  // text pixels get background reconstructed from context.
  // Pasting the full region (not just masked pixels) gives a seamless result
  // with no mismatched spots on the background.
  for (let row = 0; row < rh; row++) {
    const gy = ry1 + row
    for (let col = 0; col < rw; col++) {
      const gx = rx1 + col
      if (gx < bx || gx > bx2 || gy < by || gy > by2) continue
      const globalIdx = (gy * W + gx) * 4
      const localIdx  = (row * rw + col) * 4
      outData[globalIdx]     = resultPixels[localIdx]
      outData[globalIdx + 1] = resultPixels[localIdx + 1]
      outData[globalIdx + 2] = resultPixels[localIdx + 2]
      outData[globalIdx + 3] = 255
    }
  }
}

// ── Main pass ──────────────────────────────────────────────────────────────────

interface RectPct { x: number; y: number; w: number; h: number }
interface BubbleMsg { id: string; rect: RectPct; shape?: string; inpaint_color?: string; is_background?: boolean; rotation?: number }
interface ExpandedRect { id: string; rect: RectPct; fillColor?: string }

async function processAll(
  imageBlob: Blob,
  bubbles: BubbleMsg[],
  textMask?: Uint8Array,
): Promise<{ blob: Blob; expandedRects: ExpandedRect[] }> {
  const bitmap = await self.createImageBitmap(imageBlob)
  const W = bitmap.width
  const H = bitmap.height

  const srcCanvas = new OffscreenCanvas(W, H)
  const srcCtx    = srcCanvas.getContext('2d')!
  srcCtx.drawImage(bitmap, 0, 0)
  bitmap.close()

  const origPixels = srcCtx.getImageData(0, 0, W, H).data

  // Output buffer — transparent; flood-fill writes white, LaMa writes reconstructed pixels
  const outData = new Uint8ClampedArray(W * H * 4)

  // Route per bubble: sample 5×5 grid inside the tight text rect.
  // Speech bubble interior is white behind the strokes → high bright ratio → white fill.
  // Background text on artwork has dark/colored background → low bright ratio → LaMa.
  type Route = 'white' | 'solid' | 'lama'
  const dbg: object[] = []
  const solidColors: Map<number, { r: number; g: number; b: number; dilation: number }> = new Map()
  const fullBoxFill = new Set<number>() // bubbles that need full-box solid fill (grey screentone detected)
  const routes: Route[] = bubbles.map((b, i) => {
    const tx1 = Math.floor((b.rect.x / 100) * W)
    const ty1 = Math.floor((b.rect.y / 100) * H)
    const tx2 = Math.ceil((b.rect.x + b.rect.w) / 100 * W)
    const ty2 = Math.ceil((b.rect.y + b.rect.h) / 100 * H)
    let route: Route
    let dbgExtra: object = {}
    if (b.inpaint_color) {
      const hex = b.inpaint_color.replace('#', '')
      const r = parseInt(hex.slice(0, 2), 16)
      const g = parseInt(hex.slice(2, 4), 16)
      const bv = parseInt(hex.slice(4, 6), 16)
      solidColors.set(i, { r, g, b: bv, dilation: 3 })
      dbg.push({ bubble_no: i + 1, id: b.id, shape: b.shape, is_background: b.is_background, route: 'solid', color_override: b.inpaint_color, fill_rgb: [parseInt(b.inpaint_color.slice(1,3),16), parseInt(b.inpaint_color.slice(3,5),16), parseInt(b.inpaint_color.slice(5,7),16)], rect_pct: b.rect })
      return 'solid'
    }
    if (b.is_background === true) {
      // User-forced background route: sample color and fill solid, skip auto-detect
      const color = textMask
        ? sampleBackgroundFromMask(origPixels, W, H, tx1, ty1, tx2, ty2, textMask)
        : sampleBorderColor(origPixels, W, H, tx1, ty1, tx2, ty2)
      const dilation = textMask ? (color as ReturnType<typeof sampleBackgroundFromMask>).dilation : 3
      solidColors.set(i, { r: color.r, g: color.g, b: color.b, dilation })
      dbg.push({ bubble_no: i + 1, id: b.id, shape: b.shape, is_background: b.is_background, route: 'solid', forced_background: true, fill_rgb: [color.r, color.g, color.b], rect_pct: b.rect })
      return 'solid'
    }
    if (b.is_background === false || b.shape === 'bubble') {
      // Bubble shape always uses white/oval fill — dilation fill would be rectangular.
      // is_background===false means user explicitly marked it as a speech bubble.
      dbg.push({ bubble_no: i + 1, id: b.id, shape: b.shape, is_background: b.is_background, route: 'white', forced_white: true, rect_pct: b.rect })
      return 'white'
    }
    if (textMask) {
      const bg = sampleBackgroundFromMask(origPixels, W, H, tx1, ty1, tx2, ty2, textMask)
      const whiteCheckTriggered = bg.r > 240 && bg.g > 240 && bg.b > 240
      dbgExtra = {
        mask_used: true,
        halo_band_rgb: [bg.r, bg.g, bg.b],
        halo_band_lum: +bg.lum.toFixed(1),
        halo_band_solid: bg.solid,
        rough_bg_lum: +bg.rough_bg_lum.toFixed(1),
        halo_dilation: bg.halo_dilation,
        white_check_triggered: whiteCheckTriggered,
      }
      if (whiteCheckTriggered) {
        // Verify by sampling ALL interior pixels (including textMask-flagged ones).
        // Screentone dots are flagged as text by the mask, so non-text sampling returns
        // near-white even on grey screentone backgrounds. Including all pixels gives
        // the true perceived luminance: screentone 20% dots → mean ~204, 30% → ~178.
        let allLumSum = 0, allLumCount = 0
        for (let y = ty1; y <= ty2; y += 3)
          for (let x = tx1; x <= tx2; x += 3) {
            const p = (y * W + x) * 4
            allLumSum += origPixels[p] * 0.299 + origPixels[p + 1] * 0.587 + origPixels[p + 2] * 0.114
            allLumCount++
          }
        const allInteriorLum = allLumCount > 0 ? +(allLumSum / allLumCount).toFixed(1) : null
        dbgExtra = { ...dbgExtra, interior_px_count: allLumCount, interior_lum_avg: allInteriorLum }
        if (allLumCount > 0 && allLumSum / allLumCount < 215) {
          // Interior mean is below white threshold — could be screentone grey or dense text on white.
          // Disambiguate with a narrow ring (6px) just outside the text rect, sampling ALL pixels
          // but excluding very dark ones (bubble border / artwork outside = lum < 50).
          // White bubble: ring is white interior (mean ≈ 255).
          // Grey screentone: ring has screentone pattern (mean ≈ 180–220).
          const GREY_RING = 6
          let ringR = 0, ringG = 0, ringB = 0, ringN = 0
          const rx1 = Math.max(0, tx1 - GREY_RING), ry1 = Math.max(0, ty1 - GREY_RING)
          const rx2 = Math.min(W - 1, tx2 + GREY_RING), ry2 = Math.min(H - 1, ty2 + GREY_RING)
          for (let y = ry1; y <= ry2; y += 2)
            for (let x = rx1; x <= rx2; x += 2) {
              if (x >= tx1 && x <= tx2 && y >= ty1 && y <= ty2) continue
              const p = (y * W + x) * 4
              const lum = origPixels[p] * 0.299 + origPixels[p + 1] * 0.587 + origPixels[p + 2] * 0.114
              if (lum < 50) continue  // exclude bubble border / dark artwork
              ringR += origPixels[p]; ringG += origPixels[p + 1]; ringB += origPixels[p + 2]; ringN++
            }
          const ringMean = ringN > 0 ? (ringR * 0.299 + ringG * 0.587 + ringB * 0.114) / ringN : 255
          dbgExtra = { ...dbgExtra, ring_mean: +ringMean.toFixed(1), ring_px_count: ringN }
          if (ringMean < 220) {
            // Ring is grey — confirmed screentone/grey bubble. Fill entire box with ring color.
            const cr = ringN > 0 ? Math.round(ringR / ringN) : 128
            const cg = ringN > 0 ? Math.round(ringG / ringN) : 128
            const cb = ringN > 0 ? Math.round(ringB / ringN) : 128
            solidColors.set(i, { r: cr, g: cg, b: cb, dilation: bg.dilation })
            fullBoxFill.add(i)
            dbgExtra = { ...dbgExtra, interior_override: true, interior_rgb: [cr, cg, cb], fill_rgb: [cr, cg, cb] }
            route = 'solid'
          } else {
            // Ring is white — dark interior pixels are text strokes, not screentone. Route white.
            dbgExtra = { ...dbgExtra, interior_override: false }
            route = 'white'
          }
        } else {
          dbgExtra = { ...dbgExtra, interior_override: false }
          route = 'white'
        }
      } else if (bg.lum > 180 || bg.solid || !LAMA_ENABLED) {
        solidColors.set(i, { r: bg.r, g: bg.g, b: bg.b, dilation: bg.dilation })
        dbgExtra = { ...dbgExtra, fill_rgb: [bg.r, bg.g, bg.b] }
        route = 'solid'
      } else {
        route = 'lama'
      }
    } else {
      const border = sampleBorderColor(origPixels, W, H, tx1, ty1, tx2, ty2)
      const bright = isBrightRegion(origPixels, W, H, tx1, ty1, tx2, ty2)
      dbgExtra = { mask_used: false, bright, border_rgb: [border.r, border.g, border.b] }
      // With LaMa disabled, bright interior = speech bubble → white route (ray sampling
      // determines the actual fill color at fill time). Dark interior = background text
      // on artwork → solid fill with sampled border color.
      if (bright) {
        route = 'white'
      } else if (border.solid || !LAMA_ENABLED) {
        solidColors.set(i, { ...border, dilation: 3 })
        dbgExtra = { ...dbgExtra, fill_rgb: [border.r, border.g, border.b] }
        route = 'solid'
      } else {
        route = 'lama'
      }
    }
    dbg.push({ bubble_no: i + 1, id: b.id, shape: b.shape, is_background: b.is_background, route, rect_pct: b.rect, ...dbgExtra })
    return route
  })

  let sess: ort.InferenceSession | null = null
  if (routes.some(r => r === 'lama')) sess = await getSession()

  const expandedRects: ExpandedRect[] = []

  for (let i = 0; i < bubbles.length; i++) {
    const b   = bubbles[i]
    const tx1 = Math.floor((b.rect.x / 100) * W)
    const ty1 = Math.floor((b.rect.y / 100) * H)
    const tx2 = Math.ceil((b.rect.x + b.rect.w) / 100 * W)
    const ty2 = Math.ceil((b.rect.y + b.rect.h) / 100 * H)

    if (routes[i] === 'white') {
      // ── Speech bubble → sample interior color, then fill ──
      post({ type: 'progress', current: i, total: bubbles.length,
        stage: `Cleaning bubble ${i + 1}/${bubbles.length}…` })
      const [bx, by, bx2, by2] = scanBubbleBounds(origPixels, W, H, tx1, ty1, tx2, ty2)

      // Ray-based background color sampling.
      // Shoot 9 rays outward from each edge of the text rect (36 total).
      // For each ray, collect pixels until hitting a dark border or image edge.
      // Accept only rays with low luminance variance — uniform color means clean
      // bubble interior. High-variance rays (e.g. going through a speech tail into
      // artwork) are rejected to avoid contaminating the fill color.
      // Dark bubbles skip the dark-border stop since their interior IS dark.
      const RAY_SAMPLES = 9
      const RAY_MIN_LEN = 3
      const RAY_MAX_VAR = 600  // lum variance threshold (≈ stddev 24)
      const BORDER_DARK  = 50

      // Quick raw mean to distinguish light vs dark bubble
      let rawR = 0, rawG = 0, rawB = 0, rawN = 0
      for (let y = ty1; y <= ty2; y += 4)
        for (let x = tx1; x <= tx2; x += 4) {
          const p = (y * W + x) * 4
          rawR += origPixels[p]; rawG += origPixels[p+1]; rawB += origPixels[p+2]; rawN++
        }
      const rawMean = rawN > 0 ? (rawR * 0.299 + rawG * 0.587 + rawB * 0.114) / rawN : 128
      const isLight = rawMean > 128

      // [edgeCoord, step, perpStart, perpEnd, horizontal]
      const edges: [number, number, number, number, boolean][] = [
        [tx1 - 1, -1, ty1, ty2, true ],  // left
        [tx2 + 1,  1, ty1, ty2, true ],  // right
        [ty1 - 1, -1, tx1, tx2, false],  // top
        [ty2 + 1,  1, tx1, tx2, false],  // bottom
      ]
      let rR = 0, rG = 0, rB = 0, rN = 0
      const edgeNames = ['left', 'right', 'top', 'bottom']
      const edgeDbg: object[] = []
      for (let ei = 0; ei < edges.length; ei++) {
        const [start, step, perpS, perpE, horiz] = edges[ei]
        let edgeAccepted = 0, edgeRejectedShort = 0, edgeRejectedVar = 0
        for (let s = 0; s < RAY_SAMPLES; s++) {
          const perp = Math.round(perpS + (perpE - perpS) * s / (RAY_SAMPLES - 1))
          const lums: number[] = []
          let tR = 0, tG = 0, tB = 0
          for (let d = 0; d < MAX_EXPAND; d++) {
            const coord = start + step * d
            const x = horiz ? coord : perp
            const y = horiz ? perp  : coord
            if (x < 0 || x >= W || y < 0 || y >= H) break
            const p = (y * W + x) * 4
            const lum = origPixels[p] * 0.299 + origPixels[p+1] * 0.587 + origPixels[p+2] * 0.114
            if (isLight && lum < BORDER_DARK) break
            lums.push(lum)
            tR += origPixels[p]; tG += origPixels[p+1]; tB += origPixels[p+2]
          }
          if (lums.length < RAY_MIN_LEN) { edgeRejectedShort++; continue }
          const meanL = lums.reduce((s, v) => s + v, 0) / lums.length
          const variance = lums.reduce((s, v) => s + (v - meanL) ** 2, 0) / lums.length
          if (variance > RAY_MAX_VAR) { edgeRejectedVar++; continue }
          rR += tR; rG += tG; rB += tB; rN += lums.length
          edgeAccepted++
        }
        edgeDbg.push({ edge: edgeNames[ei], accepted: edgeAccepted, rejected_short: edgeRejectedShort, rejected_var: edgeRejectedVar })
      }
      const fallback = isLight ? 255 : 0
      const fr = rN >= 16 ? Math.round(rR / rN) : fallback
      const fg = rN >= 16 ? Math.round(rG / rN) : fallback
      const fb = rN >= 16 ? Math.round(rB / rN) : fallback

      // For bubble shape: fill full bubble interior oval (bx..bx2).
      // For rect shape: fill expanded text rect clamped to bubble bounds.
      const ex1 = b.shape === 'bubble' ? bx : Math.max(bx, tx1 - WHITE_EXPAND)
      const ey1 = b.shape === 'bubble' ? by : Math.max(by, ty1 - WHITE_EXPAND)
      const ex2 = b.shape === 'bubble' ? bx2 : Math.min(bx2, tx2 + WHITE_EXPAND)
      const ey2 = b.shape === 'bubble' ? by2 : Math.min(by2, ty2 + WHITE_EXPAND)
      const ra = (ex2 - ex1) / 2, rb = (ey2 - ey1) / 2
      const fcx = (ex1 + ex2) / 2, fcy = (ey1 + ey2) / 2
      const ang = b.rotation ? -b.rotation * Math.PI / 180 : 0
      const cosA = Math.cos(ang), sinA = Math.sin(ang)
      // Expand bounding box to cover rotated region
      const hx = ra * Math.abs(cosA) + rb * Math.abs(sinA)
      const hy = ra * Math.abs(sinA) + rb * Math.abs(cosA)
      const sx1 = Math.max(0,     Math.floor(fcx - hx))
      const sy1 = Math.max(0,     Math.floor(fcy - hy))
      const sx2 = Math.min(W - 1, Math.ceil(fcx  + hx))
      const sy2 = Math.min(H - 1, Math.ceil(fcy  + hy))
      for (let y = sy1; y <= sy2; y++) {
        for (let x = sx1; x <= sx2; x++) {
          const dx = x - fcx, dy = y - fcy
          const lx = dx * cosA - dy * sinA
          const ly = dx * sinA + dy * cosA
          // Use ellipse for bubble shape, rect for everything else
          if (b.shape === 'bubble') {
            if ((lx / ra) * (lx / ra) + (ly / rb) * (ly / rb) > 1) continue
          } else {
            if (Math.abs(lx) > ra || Math.abs(ly) > rb) continue
          }
          const idx = (y * W + x) * 4
          outData[idx] = fr; outData[idx + 1] = fg; outData[idx + 2] = fb; outData[idx + 3] = 255
        }
      }
      // Update routing debug entry with fill diagnostics
      const dbgEntry = dbg.find((d: Record<string, unknown>) => d.bubble_no === i + 1)
      if (dbgEntry) Object.assign(dbgEntry, {
        raw_mean: +rawMean.toFixed(1),
        is_light: isLight,
        accepted_ray_px: rN,
        fill_rgb: [fr, fg, fb],
        edges: edgeDbg,
      })
      expandedRects.push({
        id: b.id,
        rect: {
          x: (bx / W) * 100,
          y: (by / H) * 100,
          w: ((bx2 - bx) / W) * 100,
          h: ((by2 - by) / H) * 100,
        },
        fillColor: `#${fr.toString(16).padStart(2, '0')}${fg.toString(16).padStart(2, '0')}${fb.toString(16).padStart(2, '0')}`,
      })
    } else if (routes[i] === 'solid') {
      // ── Solid background → fill with sampled color ──
      post({ type: 'progress', current: i, total: bubbles.length,
        stage: `Cleaning background text ${i + 1}/${bubbles.length} (solid fill)…` })

      if (b.is_background === true || fullBoxFill.has(i)) {
        // Full-box fill: flood the entire box with the background color.
        // For is_background=true: sample color from ring outside the box.
        // For fullBoxFill (auto-detected grey/screentone): color already in solidColors from routing.
        let fr: number, fg: number, fb: number
        if (fullBoxFill.has(i)) {
          const sc = solidColors.get(i)!
          fr = sc.r; fg = sc.g; fb = sc.b
        } else {
          const ringFreq = new Map<number, { count: number; r: number; g: number; b: number }>()
          const rx1 = Math.max(0, tx1 - SOLID_RING), ry1 = Math.max(0, ty1 - SOLID_RING)
          const rx2 = Math.min(W - 1, tx2 + SOLID_RING), ry2 = Math.min(H - 1, ty2 + SOLID_RING)
          for (let y = ry1; y <= ry2; y += 2) {
            for (let x = rx1; x <= rx2; x += 2) {
              if (x >= tx1 && x <= tx2 && y >= ty1 && y <= ty2) continue
              if (textMask && textMask[y * W + x] > 0) continue
              const p = (y * W + x) * 4
              const r = origPixels[p], g = origPixels[p + 1], bv = origPixels[p + 2]
              const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (bv >> 3)
              const entry = ringFreq.get(key)
              if (entry) { entry.count++; entry.r += r; entry.g += g; entry.b += bv }
              else ringFreq.set(key, { count: 1, r, g, b: bv })
            }
          }
          let best = { count: 0, r: 0, g: 0, b: 0 }
          for (const entry of ringFreq.values()) if (entry.count > best.count) best = entry
          fr = best.count > 0 ? Math.round(best.r / best.count) : 0
          fg = best.count > 0 ? Math.round(best.g / best.count) : 0
          fb = best.count > 0 ? Math.round(best.b / best.count) : 0
        }
        if (b.rotation) {
          const ang = -b.rotation * Math.PI / 180
          const cosA = Math.cos(ang), sinA = Math.sin(ang)
          const rcx = (tx1 + tx2) / 2, rcy = (ty1 + ty2) / 2
          const rw = (tx2 - tx1) / 2, rh = (ty2 - ty1) / 2
          const ehx = rw * Math.abs(cosA) + rh * Math.abs(sinA)
          const ehy = rw * Math.abs(sinA) + rh * Math.abs(cosA)
          for (let y = Math.max(0, Math.floor(rcy - ehy)); y <= Math.min(H - 1, Math.ceil(rcy + ehy)); y++)
            for (let x = Math.max(0, Math.floor(rcx - ehx)); x <= Math.min(W - 1, Math.ceil(rcx + ehx)); x++) {
              const dx = x - rcx, dy = y - rcy
              const lx = dx * cosA - dy * sinA, ly = dx * sinA + dy * cosA
              if (b.shape === 'bubble') {
                if ((lx / rw) * (lx / rw) + (ly / rh) * (ly / rh) > 1) continue
              } else {
                if (Math.abs(lx) > rw || Math.abs(ly) > rh) continue
              }
              const idx = (y * W + x) * 4
              outData[idx] = fr; outData[idx + 1] = fg; outData[idx + 2] = fb; outData[idx + 3] = 255
            }
        } else if (b.shape === 'bubble') {
          const ra = (tx2 - tx1) / 2, rb = (ty2 - ty1) / 2
          const ecx = (tx1 + tx2) / 2, ecy = (ty1 + ty2) / 2
          for (let y = ty1; y <= ty2; y++)
            for (let x = tx1; x <= tx2; x++) {
              const lx = x - ecx, ly = y - ecy
              if ((lx / ra) * (lx / ra) + (ly / rb) * (ly / rb) > 1) continue
              const idx = (y * W + x) * 4
              outData[idx] = fr; outData[idx + 1] = fg; outData[idx + 2] = fb; outData[idx + 3] = 255
            }
        } else {
          for (let y = ty1; y <= ty2; y++)
            for (let x = tx1; x <= tx2; x++) {
              const idx = (y * W + x) * 4
              outData[idx] = fr; outData[idx + 1] = fg; outData[idx + 2] = fb; outData[idx + 3] = 255
            }
        }
      } else {
        // Auto-detected solid: dilate textMask and fill with pre-sampled color.
        const { r, g, b: bl, dilation: HALO_DILATION } = solidColors.get(i)!
        const fillRect = (x1: number, y1: number, x2: number, y2: number) => {
          for (let y = y1; y <= y2; y++)
            for (let x = x1; x <= x2; x++) {
              const idx = (y * W + x) * 4
              outData[idx] = r; outData[idx + 1] = g; outData[idx + 2] = bl; outData[idx + 3] = 255
            }
        }
        if (textMask) {
          const fx1 = Math.max(0, tx1 - BG_PADDING - HALO_DILATION)
          const fy1 = Math.max(0, ty1 - BG_PADDING - HALO_DILATION)
          const fx2 = Math.min(W - 1, tx2 + BG_PADDING + HALO_DILATION)
          const fy2 = Math.min(H - 1, ty2 + BG_PADDING + HALO_DILATION)
          let wrote = false
          for (let y = fy1; y <= fy2; y++) {
            for (let x = fx1; x <= fx2; x++) {
              let hit = false
              for (let dy = -HALO_DILATION; dy <= HALO_DILATION && !hit; dy++)
                for (let dx = -HALO_DILATION; dx <= HALO_DILATION && !hit; dx++) {
                  const ny = y + dy, nx = x + dx
                  if (ny >= 0 && ny < H && nx >= 0 && nx < W && textMask[ny * W + nx] > 0) hit = true
                }
              if (!hit) continue
              const idx = (y * W + x) * 4
              outData[idx] = r; outData[idx + 1] = g; outData[idx + 2] = bl; outData[idx + 3] = 255
              wrote = true
            }
          }
          if (!wrote) fillRect(fx1, fy1, fx2, fy2)
        } else {
          const sx1 = Math.max(0, tx1 - BG_PADDING), sy1 = Math.max(0, ty1 - BG_PADDING)
          const sx2 = Math.min(W - 1, tx2 + BG_PADDING), sy2 = Math.min(H - 1, ty2 + BG_PADDING)
          if (b.rotation) {
            const ang2 = -b.rotation * Math.PI / 180
            const cos2 = Math.cos(ang2), sin2 = Math.sin(ang2)
            const rcx = (sx1 + sx2) / 2, rcy = (sy1 + sy2) / 2
            const rw = (sx2 - sx1) / 2, rh = (sy2 - sy1) / 2
            const ehx = rw * Math.abs(cos2) + rh * Math.abs(sin2)
            const ehy = rw * Math.abs(sin2) + rh * Math.abs(cos2)
            for (let y = Math.max(0, Math.floor(rcy - ehy)); y <= Math.min(H - 1, Math.ceil(rcy + ehy)); y++)
              for (let x = Math.max(0, Math.floor(rcx - ehx)); x <= Math.min(W - 1, Math.ceil(rcx + ehx)); x++) {
                const dx = x - rcx, dy = y - rcy
                const lx = dx * cos2 - dy * sin2, ly = dx * sin2 + dy * cos2
                if (Math.abs(lx) > rw || Math.abs(ly) > rh) continue
                const idx = (y * W + x) * 4
                outData[idx] = r; outData[idx + 1] = g; outData[idx + 2] = bl; outData[idx + 3] = 255
              }
          } else {
            fillRect(sx1, sy1, sx2, sy2)
          }
        }
      }
    } else {
      // ── Background text → LaMa ──
      post({ type: 'progress', current: i, total: bubbles.length,
        stage: `Cleaning background text ${i + 1}/${bubbles.length} (LaMa)…` })
      const bounds: [number, number, number, number] = [
        Math.max(0, tx1 - BG_PADDING), Math.max(0, ty1 - BG_PADDING),
        Math.min(W, tx2 + BG_PADDING), Math.min(H, ty2 + BG_PADDING),
      ]
      await inpaintBackground(sess!, origPixels, outData, W, H, bounds, textMask)
    }
  }

  const outCanvas = new OffscreenCanvas(W, H)
  outCanvas.getContext('2d')!.putImageData(new ImageData(outData, W, H), 0, 0)
  const blob = await outCanvas.convertToBlob({ type: 'image/png' })
  post({ type: 'debug', data: { bubbles: dbg } })
  return { blob, expandedRects }
}

// ── Message handler ────────────────────────────────────────────────────────────

self.addEventListener('message', async (e: MessageEvent) => {
  const msg = e.data
  if (msg.type !== 'inpaint') return
  try {
    post({ type: 'progress', current: 0, total: msg.bubbles.length, stage: 'Starting…' })
    const textMask = msg.textMask instanceof Uint8Array ? msg.textMask as Uint8Array : undefined
    const { blob, expandedRects } = await processAll(msg.imageBlob as Blob, msg.bubbles as BubbleMsg[], textMask)
    post({ type: 'done', resultBlob: blob, expandedRects })
  } catch (err) {
    post({ type: 'error', message: String(err) })
  }
})
