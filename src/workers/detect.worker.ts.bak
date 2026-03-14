/**
 * detect.worker.ts
 *
 * Runs the comic-text-detector ONNX model entirely inside a Web Worker.
 *
 * Model source: https://huggingface.co/mayocream/comic-text-detector-onnx
 * Architecture: YOLOv5s backbone + UnetHead + DBHead — all fused into one ONNX.
 * We only consume the `blk` output (YOLO text block detections).
 *
 * After YOLO detection, each tight text-bounding-box is expanded by scanning
 * outward in 4 directions through the original image pixels until a dark pixel
 * (bubble border) is hit. This gives an inpaint/typeset rect that fills the
 * interior of the speech bubble rather than just the text itself.
 *
 * Message protocol:
 *   IN  { type: 'detect', imageBlob: Blob }
 *   OUT { type: 'progress', stage: string, value: number }   // 0–1
 *   OUT { type: 'result',   bubbles: MangaBubble[] }
 *   OUT { type: 'error',    message: string }
 */

import * as ort from 'onnxruntime-web'
import type { MangaBubble } from '../types'

// ── Config ────────────────────────────────────────────────────────────────────

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/'

const MODEL_URL =
  'https://huggingface.co/mayocream/comic-text-detector-onnx/resolve/main/comic-text-detector.onnx'

const INPUT_SIZE   = 1024   // model expects 1024×1024 input
const CONF_THRESH  = 0.45   // minimum object × class confidence
const IOU_THRESH   = 0.3    // NMS IoU threshold

// ── Double-bubble split config ────────────────────────────────────────────────

const SPLIT_MIN_H      = 40   // minimum height/width (px orig) for each sub-bubble after split
const MASK_TEXT_THRESH = 0.3  // mask value above this = text pixel
const MIN_GAP_FRAC     = 0.15 // largest gap must be ≥ 15% of box dimension
const MIN_DOMINANCE    = 2.5  // largest gap must be ≥ 2.5× the second-largest gap
const MIN_VOTE_FRAC    = 0.20 // fraction of slices that must agree on a gap

// ── False-positive split rejection ────────────────────────────────────────────
// After tightenToMask on each split half, reject the split when BOTH:
//   1. Gap between text clusters < SEAM_GAP_FRAC × total box dimension — text nearly touches the seam
//   2. Text clusters overlap >= SEAM_OVERLAP_FRAC on the axis perpendicular to the cut:
//      seamY (horizontal cut) → check X-overlap; seamX (vertical cut) → check Y-overlap
//      High perpendicular overlap means both clusters sit in the same column/row → one bubble.
// This catches single bubbles whose large inter-line gap was mistaken for a seam.

const SEAM_GAP_FRAC     = 0.20  // reject split if text-cluster gap < 20% of total box (was 0.08)
const SEAM_OVERLAP_FRAC = 0.80  // reject split if perpendicular-axis overlap >= 80% of narrower cluster

// ── Singleton session ─────────────────────────────────────────────────────────

let session: ort.InferenceSession | null = null

async function getSession(): Promise<ort.InferenceSession> {
  if (session) return session

  progress('Downloading model (90 MB, cached after first run)…', 0.05)

  const resp = await fetch(MODEL_URL)
  if (!resp.ok) throw new Error(`Model fetch failed: ${resp.status} ${resp.statusText}`)

  const total = Number(resp.headers.get('content-length') ?? 0)
  const reader = resp.body!.getReader()
  const chunks: Uint8Array[] = []
  let received = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    if (total > 0) progress('Downloading model…', 0.05 + 0.55 * (received / total))
  }

  const buffer = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length }

  progress('Loading model into ONNX runtime…', 0.62)
  session = await ort.InferenceSession.create(buffer.buffer, {
    executionProviders: ['wasm'],
  })

  return session
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

interface Preprocessed {
  tensor: ort.Tensor
  scale:  number
  dw:     number
  dh:     number
  origW:  number
  origH:  number
}

async function preprocess(blob: Blob): Promise<Preprocessed> {
  const bitmap = await createImageBitmap(blob)
  const { width: origW, height: origH } = bitmap

  // Letterbox to 1024×1024 for YOLO input
  const scale = Math.min(INPUT_SIZE / origW, INPUT_SIZE / origH)
  const newW  = Math.round(origW * scale)
  const newH  = Math.round(origH * scale)
  const dw    = Math.floor((INPUT_SIZE - newW) / 2)
  const dh    = Math.floor((INPUT_SIZE - newH) / 2)

  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE)
  const ctx    = canvas.getContext('2d')!
  ctx.fillStyle = 'rgb(114,114,114)'
  ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE)
  ctx.drawImage(bitmap, dw, dh, newW, newH)
  bitmap.close()

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
  const N = INPUT_SIZE * INPUT_SIZE
  const floats = new Float32Array(3 * N)

  for (let i = 0; i < N; i++) {
    floats[i]         = data[i * 4]     / 255
    floats[N + i]     = data[i * 4 + 1] / 255
    floats[2 * N + i] = data[i * 4 + 2] / 255
  }

  return {
    tensor: new ort.Tensor('float32', floats, [1, 3, INPUT_SIZE, INPUT_SIZE]),
    scale, dw, dh, origW, origH,
  }
}

// ── Double-bubble seam detector ───────────────────────────────────────────────

/**
 * Per-column largest-gap analysis on the model segmentation mask.
 *
 * For each X column inside the YOLO box:
 *   1. Walk all rows top→bottom, recording gaps between consecutive text runs.
 *   2. Find the LARGEST gap whose centre falls in the middle 20–80% of the box.
 *   3. The gap must be ≥ MIN_GAP_FRAC × boxH (absolute size filter).
 *   4. The gap must be ≥ MIN_DOMINANCE × second-largest gap (uniqueness filter).
 *      This rejects single bubbles where all inter-char gaps are similar in size.
 *   5. Columns that pass both filters vote for a split at their largest-gap centre.
 *
 * If ≥ MIN_VOTE_FRAC of columns vote, the median vote is the seam Y.
 */
function findSeamY(
  maskData: Float32Array,
  maskW:    number,
  bx1: number, by1: number,
  bx2: number, by2: number,
  scale: number, dh: number,
  origH: number,
): number | null {
  const ix1 = Math.max(0,          Math.round(bx1))
  const iy1 = Math.max(0,          Math.round(by1))
  const ix2 = Math.min(maskW - 1,  Math.round(bx2))
  const iy2 = Math.min(maskW - 1,  Math.round(by2))

  const boxH = iy2 - iy1
  const boxW = ix2 - ix1
  if (boxH < 20 || boxW <= 0) return null

  const marginLo  = iy1 + Math.round(boxH * 0.20)
  const marginHi  = iy2 - Math.round(boxH * 0.20)
  const minGapRows = boxH * MIN_GAP_FRAC

  const gapCenters: number[] = []

  for (let col = ix1; col <= ix2; col++) {
    // Collect all gaps between text runs in this column
    const gaps: { size: number; center: number }[] = []
    let prevTextEnd = -1

    for (let row = iy1; row <= iy2; row++) {
      if (maskData[row * maskW + col] > MASK_TEXT_THRESH) {
        if (prevTextEnd >= 0) {
          const gapSize   = row - prevTextEnd - 1
          const gapCenter = (prevTextEnd + row) / 2
          if (gapSize > 0 && gapCenter >= marginLo && gapCenter <= marginHi) {
            gaps.push({ size: gapSize, center: gapCenter })
          }
        }
        prevTextEnd = row
      }
    }

    if (gaps.length === 0) continue

    // Sort largest first
    gaps.sort((a, b) => b.size - a.size)
    const largest    = gaps[0]
    const secondSize = gaps[1]?.size ?? 0

    // Absolute size filter
    if (largest.size < minGapRows) continue

    // Dominance filter: largest must be clearly bigger than all others
    if (secondSize > 0 && largest.size < secondSize * MIN_DOMINANCE) continue

    gapCenters.push(largest.center)
  }

  // Need enough columns agreeing on a gap
  if (gapCenters.length < boxW * MIN_VOTE_FRAC) return null

  // Seam = median vote, converted to original image coords
  gapCenters.sort((a, b) => a - b)
  const seamMaskY = gapCenters[Math.floor(gapCenters.length / 2)]
  const seamOrigY = (seamMaskY - dh) / scale

  // Both resulting halves must be tall enough
  const origY1 = Math.max(0,     (iy1 - dh) / scale)
  const origY2 = Math.min(origH, (iy2 - dh) / scale)
  if (seamOrigY - origY1 < SPLIT_MIN_H || origY2 - seamOrigY < SPLIT_MIN_H) return null

  return Math.max(0, Math.min(origH - 1, seamOrigY))
}

/**
 * Same logic as findSeamY but transposed: scans each ROW for horizontal gaps
 * between text runs to detect a left-right double bubble.
 * Returns the seam X in original image pixel coords, or null.
 */
function findSeamX(
  maskData: Float32Array,
  maskW:    number,
  bx1: number, by1: number,
  bx2: number, by2: number,
  scale: number, dw: number,
  origW: number,
): number | null {
  const ix1 = Math.max(0,          Math.round(bx1))
  const iy1 = Math.max(0,          Math.round(by1))
  const ix2 = Math.min(maskW - 1,  Math.round(bx2))
  const iy2 = Math.min(maskW - 1,  Math.round(by2))

  const boxW = ix2 - ix1
  const boxH = iy2 - iy1
  if (boxW < 20 || boxH <= 0) return null

  const marginLo   = ix1 + Math.round(boxW * 0.20)
  const marginHi   = ix2 - Math.round(boxW * 0.20)
  const minGapCols = boxW * MIN_GAP_FRAC

  const gapCenters: number[] = []

  for (let row = iy1; row <= iy2; row++) {
    const gaps: { size: number; center: number }[] = []
    let prevTextEnd = -1

    for (let col = ix1; col <= ix2; col++) {
      if (maskData[row * maskW + col] > MASK_TEXT_THRESH) {
        if (prevTextEnd >= 0) {
          const gapSize   = col - prevTextEnd - 1
          const gapCenter = (prevTextEnd + col) / 2
          if (gapSize > 0 && gapCenter >= marginLo && gapCenter <= marginHi) {
            gaps.push({ size: gapSize, center: gapCenter })
          }
        }
        prevTextEnd = col
      }
    }

    if (gaps.length === 0) continue

    gaps.sort((a, b) => b.size - a.size)
    const largest    = gaps[0]
    const secondSize = gaps[1]?.size ?? 0

    if (largest.size < minGapCols) continue
    if (secondSize > 0 && largest.size < secondSize * MIN_DOMINANCE) continue

    gapCenters.push(largest.center)
  }

  if (gapCenters.length < boxH * MIN_VOTE_FRAC) return null

  gapCenters.sort((a, b) => a - b)
  const seamMaskX = gapCenters[Math.floor(gapCenters.length / 2)]
  const seamOrigX = (seamMaskX - dw) / scale

  const origX1 = Math.max(0,     (ix1 - dw) / scale)
  const origX2 = Math.min(origW, (ix2 - dw) / scale)
  if (seamOrigX - origX1 < SPLIT_MIN_H || origX2 - seamOrigX < SPLIT_MIN_H) return null

  return Math.max(0, Math.min(origW - 1, seamOrigX))
}

// ── Mask-based tightening ─────────────────────────────────────────────────────

function tightenToMask(
  maskData: Float32Array, maskW: number,
  ox1: number, oy1: number, ox2: number, oy2: number,
  scale: number, dw: number, dh: number,
): [number, number, number, number] | null {
  const mx1 = Math.max(0,         Math.round(ox1 * scale + dw))
  const my1 = Math.max(0,         Math.round(oy1 * scale + dh))
  const mx2 = Math.min(maskW - 1, Math.round(ox2 * scale + dw))
  const my2 = Math.min(maskW - 1, Math.round(oy2 * scale + dh))
  let minX = mx2 + 1, maxX = mx1 - 1, minY = my2 + 1, maxY = my1 - 1
  for (let row = my1; row <= my2; row++)
    for (let col = mx1; col <= mx2; col++)
      if (maskData[row * maskW + col] > MASK_TEXT_THRESH) {
        if (col < minX) minX = col; if (col > maxX) maxX = col
        if (row < minY) minY = row; if (row > maxY) maxY = row
      }
  if (minX > maxX || minY > maxY) return null
  return [
    Math.max(0, (minX - dw) / scale), Math.max(0, (minY - dh) / scale),
    (maxX - dw) / scale,              (maxY - dh) / scale,
  ]
}

// ── Post-processing (YOLO blk output) ────────────────────────────────────────

function xywh2xyxy(cx: number, cy: number, w: number, h: number): [number,number,number,number] {
  return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
}

function boxIou(
  a: [number,number,number,number],
  b: [number,number,number,number],
): number {
  const ix1 = Math.max(a[0], b[0]), iy1 = Math.max(a[1], b[1])
  const ix2 = Math.min(a[2], b[2]), iy2 = Math.min(a[3], b[3])
  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1)
  const aA = (a[2] - a[0]) * (a[3] - a[1])
  const bA = (b[2] - b[0]) * (b[3] - b[1])
  return inter / (aA + bA - inter + 1e-6)
}

function nms(
  boxes:  [number,number,number,number][],
  scores: number[],
  iouThr: number,
): number[] {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b] - scores[a])
  const suppressed = new Set<number>()
  const kept: number[] = []

  for (const i of order) {
    if (suppressed.has(i)) continue
    kept.push(i)
    for (const j of order) {
      if (i === j || suppressed.has(j)) continue
      if (boxIou(boxes[i], boxes[j]) > iouThr) suppressed.add(j)
    }
  }
  return kept
}

function processBlk(
  data:     Float32Array,
  dims:     readonly number[],
  scale:    number,
  dw:       number,
  dh:       number,
  origW:    number,
  origH:    number,
  maskData: Float32Array | null,
  maskW:    number,
): MangaBubble[] {
  const N    = dims[1]
  const cols = dims[2]

  const boxes:  [number,number,number,number][] = []
  const scores: number[] = []

  for (let i = 0; i < N; i++) {
    const base    = i * cols
    const objConf = data[base + 4]
    const clsConf = Math.max(data[base + 5], data[base + 6], data[base + 7])
    const conf    = objConf * clsConf

    if (conf < CONF_THRESH) continue

    boxes.push(xywh2xyxy(data[base], data[base + 1], data[base + 2], data[base + 3]))
    scores.push(conf)
  }

  const kept = nms(boxes, scores, IOU_THRESH)

  // ── Debug: collect per-box info, print as one entry ───────────────────────
  const dbg: object[] = []

  return kept.flatMap(idx => {
    const [x1, y1, x2, y2] = boxes[idx]

    // Unpad letterbox → original pixel coords (tight text box)
    const ox1 = Math.max(0,     (x1 - dw) / scale)
    const oy1 = Math.max(0,     (y1 - dh) / scale)
    const ox2 = Math.min(origW, (x2 - dw) / scale)
    const oy2 = Math.min(origH, (y2 - dh) / scale)

    // Skip degenerate boxes
    if (ox2 <= ox1 || oy2 <= oy1) return []

    // Check for a double-bubble seam (top-bottom first, then left-right).
    const minSplitDim = SPLIT_MIN_H * 2
    const seamY = (maskData && (ox2 - ox1) >= minSplitDim)
      ? findSeamY(maskData, maskW, x1, y1, x2, y2, scale, dh, origH)
      : null
    const seamX = (!seamY && maskData && (oy2 - oy1) >= minSplitDim)
      ? findSeamX(maskData, maskW, x1, y1, x2, y2, scale, dw, origW)
      : null

    // Output tight text rects — bubble border expansion happens at inpaint time.
    let tightRects: [number, number, number, number][]
    const boxDbg: Record<string, unknown> = {
      yolo_raw:    { x1: +x1.toFixed(1), y1: +y1.toFixed(1), x2: +x2.toFixed(1), y2: +y2.toFixed(1), score: +scores[idx].toFixed(3) },
      unpadded_px: { ox1: +ox1.toFixed(1), oy1: +oy1.toFixed(1), ox2: +ox2.toFixed(1), oy2: +oy2.toFixed(1), w: +(ox2-ox1).toFixed(1), h: +(oy2-oy1).toFixed(1) },
      seam: seamY !== null ? { axis: 'Y', at: +seamY.toFixed(1) } : seamX !== null ? { axis: 'X', at: +seamX.toFixed(1) } : 'none',
    }

    if (seamY !== null) {
      const t1 = maskData ? tightenToMask(maskData, maskW, ox1, oy1, ox2, seamY, scale, dw, dh) : null
      const t2 = maskData ? tightenToMask(maskData, maskW, ox1, seamY, ox2, oy2, scale, dw, dh) : null

      boxDbg.tight_halves = {
        t1: t1 ? { x1: +t1[0].toFixed(1), y1: +t1[1].toFixed(1), x2: +t1[2].toFixed(1), y2: +t1[3].toFixed(1) } : null,
        t2: t2 ? { x1: +t2[0].toFixed(1), y1: +t2[1].toFixed(1), x2: +t2[2].toFixed(1), y2: +t2[3].toFixed(1) } : null,
      }

      const isFalsePositive = t1 !== null && t2 !== null && (() => {
        const totalH        = oy2 - oy1
        const gapH          = t2[1] - t1[3]
        const xOverlap      = Math.max(0, Math.min(t1[2], t2[2]) - Math.max(t1[0], t2[0]))
        const minW          = Math.min(t1[2] - t1[0], t2[2] - t2[0])
        const xOverlapRatio = minW > 0 ? xOverlap / minW : 0
        const gapRatio      = gapH / totalH
        const result        = gapRatio < SEAM_GAP_FRAC && xOverlapRatio >= SEAM_OVERLAP_FRAC
        boxDbg.fp_check = { gapRatio: +gapRatio.toFixed(3), gapThresh: SEAM_GAP_FRAC, xOverlapRatio: +xOverlapRatio.toFixed(3), overlapThresh: SEAM_OVERLAP_FRAC, result }
        return result
      })()

      if (isFalsePositive) {
        const tight = maskData ? tightenToMask(maskData, maskW, ox1, oy1, ox2, oy2, scale, dw, dh) : null
        tightRects = [tight ?? [ox1, oy1, ox2, oy2]]
        boxDbg.outcome = 'reverted_to_single'
      } else {
        tightRects = [
          t1 ?? [ox1, oy1, ox2, seamY],
          t2 ?? [ox1, seamY, ox2, oy2],
        ]
        boxDbg.outcome = 'split_Y'
      }
    } else if (seamX !== null) {
      const t1 = maskData ? tightenToMask(maskData, maskW, ox1, oy1, seamX, oy2, scale, dw, dh) : null
      const t2 = maskData ? tightenToMask(maskData, maskW, seamX, oy1, ox2, oy2, scale, dw, dh) : null

      boxDbg.tight_halves = {
        t1: t1 ? { x1: +t1[0].toFixed(1), y1: +t1[1].toFixed(1), x2: +t1[2].toFixed(1), y2: +t1[3].toFixed(1) } : null,
        t2: t2 ? { x1: +t2[0].toFixed(1), y1: +t2[1].toFixed(1), x2: +t2[2].toFixed(1), y2: +t2[3].toFixed(1) } : null,
      }

      const isFalsePositive = t1 !== null && t2 !== null && (() => {
        const totalW        = ox2 - ox1
        const gapW          = t2[0] - t1[2]
        const yOverlap      = Math.max(0, Math.min(t1[3], t2[3]) - Math.max(t1[1], t2[1]))
        const minH          = Math.min(t1[3] - t1[1], t2[3] - t2[1])
        const yOverlapRatio = minH > 0 ? yOverlap / minH : 0
        const gapRatio      = gapW / totalW
        const result        = gapRatio < SEAM_GAP_FRAC && yOverlapRatio >= SEAM_OVERLAP_FRAC
        boxDbg.fp_check = { gapRatio: +gapRatio.toFixed(3), gapThresh: SEAM_GAP_FRAC, yOverlapRatio: +yOverlapRatio.toFixed(3), overlapThresh: SEAM_OVERLAP_FRAC, result }
        return result
      })()

      if (isFalsePositive) {
        const tight = maskData ? tightenToMask(maskData, maskW, ox1, oy1, ox2, oy2, scale, dw, dh) : null
        tightRects = [tight ?? [ox1, oy1, ox2, oy2]]
        boxDbg.outcome = 'reverted_to_single'
      } else {
        tightRects = [
          t1 ?? [ox1, oy1, seamX, oy2],
          t2 ?? [seamX, oy1, ox2, oy2],
        ]
        boxDbg.outcome = 'split_X'
      }
    } else {
      // Single bubble — store tight text region
      const tight = maskData ? tightenToMask(maskData, maskW, ox1, oy1, ox2, oy2, scale, dw, dh) : null
      tightRects = [tight ?? [ox1, oy1, ox2, oy2]]
      boxDbg.outcome = 'single'
    }

    boxDbg.final_pct = tightRects.map(([rx1, ry1, rx2, ry2]) => ({
      x: +((rx1/origW)*100).toFixed(2), y: +((ry1/origH)*100).toFixed(2),
      w: +(((rx2-rx1)/origW)*100).toFixed(2), h: +(((ry2-ry1)/origH)*100).toFixed(2),
    }))
    dbg.push(boxDbg)

    return tightRects.flatMap(([rx1, ry1, rx2, ry2]) => {
      if (rx2 <= rx1 || ry2 <= ry1) return []
      return [{
        id:           crypto.randomUUID(),
        rect: {
          x: (rx1 / origW) * 100,
          y: (ry1 / origH) * 100,
          w: ((rx2 - rx1) / origW) * 100,
          h: ((ry2 - ry1) / origH) * 100,
        },
        raw_ja:        '',
        translated_zh: '',
        state:         'detected' as const,
        is_locked:     false,
        layer_z:       0,
      }]
    })
  }).concat((() => { console.log('[detect]', { beforeNMS: boxes.length, afterNMS: kept.length, boxes: dbg }); return [] })())
}

// ── Post-split deduplication ──────────────────────────────────────────────────

/**
 * After splitting, three kinds of duplicates can appear:
 *   A) A small bubble that is mostly INSIDE a larger one → drop the small one.
 *   B) A large bubble whose area is mostly COVERED BY the sum of smaller ones → drop the large one.
 *   C) A large bubble that acts as a YOLO wrapper over 2+ real smaller bubbles each contained within
 *      it (e.g. YOLO fires one big box around two adjacent bubbles that were also detected
 *      individually) → drop the large wrapper.
 */
function deduplicateBubbles(bubbles: MangaBubble[]): MangaBubble[] {
  const kept: boolean[] = bubbles.map(() => true)

  // Pass 1 — drop large wrapper boxes: if 2+ smaller bubbles are each ≥70% contained
  // inside a larger bubble, the large one is a spurious YOLO detection (case C).
  // Must run before Pass 2 so the smaller bubbles are still marked kept.
  for (let i = 0; i < bubbles.length; i++) {
    if (!kept[i]) continue
    const a = bubbles[i].rect
    const aArea = a.w * a.h
    let containedCount = 0
    for (let j = 0; j < bubbles.length; j++) {
      if (i === j || !kept[j]) continue
      const b = bubbles[j].rect
      const bArea = b.w * b.h
      if (bArea >= aArea) continue  // only count smaller bubbles
      const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
      const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))
      const inter = ix * iy
      if (inter / bArea >= 0.70) containedCount++
    }
    if (containedCount >= 2) kept[i] = false
  }

  // Pass 2 — drop any bubble mostly inside a single larger bubble (case A)
  for (let i = 0; i < bubbles.length; i++) {
    if (!kept[i]) continue
    const a = bubbles[i].rect
    const aArea = a.w * a.h
    for (let j = 0; j < bubbles.length; j++) {
      if (i === j || !kept[j]) continue
      const b = bubbles[j].rect
      const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
      const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))
      const inter = ix * iy
      const bArea = b.w * b.h
      if (inter / bArea >= 0.85 && aArea > bArea) kept[j] = false
    }
  }

  // Pass 3 — drop any bubble whose area is mostly covered by others combined (case B)
  for (let i = 0; i < bubbles.length; i++) {
    if (!kept[i]) continue
    const a = bubbles[i].rect
    const aArea = a.w * a.h
    let totalCoverage = 0
    for (let j = 0; j < bubbles.length; j++) {
      if (i === j || !kept[j]) continue
      const b = bubbles[j].rect
      const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x))
      const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))
      totalCoverage += ix * iy
    }
    if (totalCoverage / aArea >= 0.85) kept[i] = false
  }

  return bubbles.filter((_, i) => kept[i])
}

// ── Worker message handler ────────────────────────────────────────────────────

function progress(stage: string, value: number): void {
  self.postMessage({ type: 'progress', stage, value })
}

self.onmessage = async (e: MessageEvent) => {
  if (e.data?.type !== 'detect') return

  try {
    const sess = await getSession()

    progress('Preprocessing image…', 0.7)
    const { tensor, scale, dw, dh, origW, origH } = await preprocess(e.data.imageBlob as Blob)

    progress('Running detection…', 0.82)
    const outputs = await sess.run({ images: tensor })

    progress('Processing detections…', 0.95)

    const blk     = outputs['blk']  ?? Object.values(outputs)[0]
    const maskOut = outputs['mask'] ?? (Object.values(outputs).length > 1 ? Object.values(outputs)[1] : null)
    const maskData = maskOut ? (maskOut.data as Float32Array) : null
    const maskW    = maskOut ? maskOut.dims[maskOut.dims.length - 1] : 0

    const bubbles = deduplicateBubbles(processBlk(
      blk.data as Float32Array,
      blk.dims,
      scale, dw, dh, origW, origH,
      maskData, maskW,
    ))

    self.postMessage({ type: 'result', bubbles })
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err) })
  }
}
