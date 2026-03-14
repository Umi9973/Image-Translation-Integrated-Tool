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
 * Model: Carve/LaMa-ONNX / lama_fp32.onnx (~208 MB, OPFS-cached after first use)
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

const MODEL_URL     = 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx'
const OPFS_FILENAME = 'lama_fp32.onnx'
const LAMA_SIZE     = 512
const LAMA_CTX_FRAC = 0.5   // context padding around background text for LaMa (fraction of max(w,h))
const BG_PADDING    = 8     // px padding around tight text rect for LaMa bounds
const SAMPLE_PAD    = 20    // px expansion around tight rect used for brightness sampling

// ── Bubble boundary scanner ────────────────────────────────────────────────────
// Used for speech bubbles only: expands the tight text rect outward until
// hitting the dark bubble border, returning the full bubble interior bounds.

const DARK_THRESH  = 80
const MAX_EXPAND   = 200
const SCAN_SAMPLES = 9
const WHITE_EXPAND = 7   // px to expand tight text rect when painting white
const MIN_MARGIN   = 4   // only expand a side if bubble border is at least this far away

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
    Math.max(0, x1 - leftExp),  Math.max(0, y1 - topExp),
    Math.min(W, x2 + rightExp), Math.min(H, y2 + botExp),
  ]
}

// ── Background brightness check ───────────────────────────────────────────────
// Samples a 5×5 grid inside bx/by/bx2/by2 but outside the tight text rect.
// Returns true when the majority are bright → speech bubble.
// Returns false when the background is dark/colored → background text on artwork.

const BRIGHT_THRESH     = 200   // luminance above this = bright pixel
const BRIGHT_MIN_RATIO  = 0.55  // fraction of samples that must be bright to confirm bubble

function isBrightRegion(
  pixels: Uint8ClampedArray,
  W: number, H: number,
  bx: number, by: number, bx2: number, by2: number,
  tx1: number, ty1: number, tx2: number, ty2: number,
): boolean {
  const GRID = 5  // 5×5 sample grid
  let bright = 0, total = 0
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const sx = Math.round(bx + (bx2 - bx) * (gx + 0.5) / GRID)
      const sy = Math.round(by + (by2 - by) * (gy + 0.5) / GRID)
      if (sx >= tx1 && sx <= tx2 && sy >= ty1 && sy <= ty2) continue  // skip text pixels
      const xi = Math.max(0, Math.min(W - 1, sx))
      const yi = Math.max(0, Math.min(H - 1, sy))
      const b  = (yi * W + xi) * 4
      const l  = pixels[b] * 0.299 + pixels[b + 1] * 0.587 + pixels[b + 2] * 0.114
      if (l > BRIGHT_THRESH) bright++
      total++
    }
  }
  return total > 0 && (bright / total) >= BRIGHT_MIN_RATIO
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
  srcCanvas.getContext('2d')!.putImageData(new ImageData(pixels, srcW, srcH), 0, 0)
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

async function getSession(): Promise<ort.InferenceSession> {
  if (session) return session

  post({ type: 'progress', current: 0, total: 1, stage: 'Checking LaMa model cache…' })
  let buffer = await loadFromOpfs()

  if (!buffer) {
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
          stage: `Downloading LaMa model… ${pct}% (208 MB, cached after first run)` })
      }
    }
    const arr = new Uint8Array(received)
    let off = 0
    for (const chunk of chunks) { arr.set(chunk, off); off += chunk.length }
    buffer = arr.buffer
    saveToOpfs(buffer)  // fire-and-forget
  }

  post({ type: 'progress', current: 0, total: 1, stage: 'Loading LaMa into ONNX runtime…' })
  session = await ort.InferenceSession.create(buffer, { executionProviders: ['wasm'] })
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
    imgData[i]         = imgPixels[i * 4]     / 255
    imgData[N + i]     = imgPixels[i * 4 + 1] / 255
    imgData[2 * N + i] = imgPixels[i * 4 + 2] / 255
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

  // Rectangular mask: cover the bounds region within the crop
  const cropMask = new Uint8Array(rw * rh)
  for (let row = 0; row < rh; row++) {
    const gy = ry1 + row
    for (let col = 0; col < rw; col++) {
      const gx = rx1 + col
      if (gx >= bx && gx <= bx2 && gy >= by && gy <= by2) cropMask[row * rw + col] = 255
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
    outRgba[p * 4]     = Math.max(0, Math.min(255, Math.round(output[p]         * 255)))
    outRgba[p * 4 + 1] = Math.max(0, Math.min(255, Math.round(output[N + p]     * 255)))
    outRgba[p * 4 + 2] = Math.max(0, Math.min(255, Math.round(output[2 * N + p] * 255)))
    outRgba[p * 4 + 3] = 255
  }
  const resultPixels = scalePixels(outRgba, LAMA_SIZE, LAMA_SIZE, rw, rh)

  // Paste only masked pixels into outData (the transparent overlay)
  for (let row = 0; row < rh; row++) {
    for (let col = 0; col < rw; col++) {
      if (cropMask[row * rw + col] === 0) continue
      const globalIdx = ((ry1 + row) * W + (rx1 + col)) * 4
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
interface BubbleMsg { id: string; rect: RectPct }
interface ExpandedRect { id: string; rect: RectPct }

async function processAll(
  imageBlob: Blob,
  bubbles: BubbleMsg[],
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

  // Route per bubble: sample around the tight text rect to decide.
  // Bright surrounding region = speech bubble → paint rect white + scan full bounds.
  // Dark/colored = background text on artwork → LaMa.
  type Route = 'white' | 'lama'
  const routes: Route[] = bubbles.map(b => {
    const tx1 = Math.floor((b.rect.x / 100) * W)
    const ty1 = Math.floor((b.rect.y / 100) * H)
    const tx2 = Math.ceil((b.rect.x + b.rect.w) / 100 * W)
    const ty2 = Math.ceil((b.rect.y + b.rect.h) / 100 * H)
    const bx  = Math.max(0, tx1 - SAMPLE_PAD)
    const by  = Math.max(0, ty1 - SAMPLE_PAD)
    const bx2 = Math.min(W, tx2 + SAMPLE_PAD)
    const by2 = Math.min(H, ty2 + SAMPLE_PAD)
    return isBrightRegion(origPixels, W, H, bx, by, bx2, by2, tx1, ty1, tx2, ty2)
      ? 'white' : 'lama'
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
      // ── Speech bubble → paint white ──
      post({ type: 'progress', current: i, total: bubbles.length,
        stage: `Cleaning bubble ${i + 1}/${bubbles.length}…` })
      // Scan full bubble interior first so we know the clearance on each side.
      const [bx, by, bx2, by2] = scanBubbleBounds(origPixels, W, H, tx1, ty1, tx2, ty2)
      // Expand each side by WHITE_EXPAND only if there's enough room before the border.
      const px1 = (tx1 - bx)  >= WHITE_EXPAND + MIN_MARGIN ? tx1 - WHITE_EXPAND : tx1
      const py1 = (ty1 - by)  >= WHITE_EXPAND + MIN_MARGIN ? ty1 - WHITE_EXPAND : ty1
      const px2 = (bx2 - tx2) >= WHITE_EXPAND + MIN_MARGIN ? tx2 + WHITE_EXPAND : tx2
      const py2 = (by2 - ty2) >= WHITE_EXPAND + MIN_MARGIN ? ty2 + WHITE_EXPAND : ty2
      for (let y = py1; y <= py2; y++) {
        for (let x = px1; x <= px2; x++) {
          const idx = (y * W + x) * 4
          outData[idx] = 255; outData[idx + 1] = 255; outData[idx + 2] = 255; outData[idx + 3] = 255
        }
      }
      expandedRects.push({
        id: b.id,
        rect: {
          x: (bx / W) * 100,
          y: (by / H) * 100,
          w: ((bx2 - bx) / W) * 100,
          h: ((by2 - by) / H) * 100,
        },
      })
    } else {
      // ── Background text → LaMa ──
      post({ type: 'progress', current: i, total: bubbles.length,
        stage: `Cleaning background text ${i + 1}/${bubbles.length} (LaMa)…` })
      const bounds: [number, number, number, number] = [
        Math.max(0, tx1 - BG_PADDING), Math.max(0, ty1 - BG_PADDING),
        Math.min(W, tx2 + BG_PADDING), Math.min(H, ty2 + BG_PADDING),
      ]
      await inpaintBackground(sess!, origPixels, outData, W, H, bounds)
    }
  }

  const outCanvas = new OffscreenCanvas(W, H)
  outCanvas.getContext('2d')!.putImageData(new ImageData(outData, W, H), 0, 0)
  const blob = await outCanvas.convertToBlob({ type: 'image/png' })
  return { blob, expandedRects }
}

// ── Message handler ────────────────────────────────────────────────────────────

self.addEventListener('message', async (e: MessageEvent) => {
  const msg = e.data
  if (msg.type !== 'inpaint') return
  try {
    post({ type: 'progress', current: 0, total: msg.bubbles.length, stage: 'Starting…' })
    const { blob, expandedRects } = await processAll(msg.imageBlob as Blob, msg.bubbles as BubbleMsg[])
    post({ type: 'done', resultBlob: blob, expandedRects })
  } catch (err) {
    post({ type: 'error', message: String(err) })
  }
})
