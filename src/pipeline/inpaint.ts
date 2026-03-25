/**
 * inpaint.ts — Inpainting pipeline stage
 *
 * Manages the inpaint Web Worker lifecycle and exposes a clean async API.
 * Heavy work (LaMa ONNX inference) runs in inpaint.worker.ts, not the main thread.
 *
 * Signature must not change — workspace.ts depends on it.
 */

import type { MangaBubble } from '../types'
import type { DetectionMask } from './detect'

// Singleton worker — created once, reused across inpainting calls
let _worker: Worker | null = null

function getWorker(): Worker {
  if (!_worker) {
    _worker = new Worker(
      new URL('../workers/inpaint.worker.ts', import.meta.url),
      { type: 'module' },
    )
  }
  return _worker
}

export interface InpaintResult {
  blob: Blob
  /** Expanded bubble interior rects (percentage-based) for speech bubbles only. */
  expandedRects: { id: string; rect: { x: number; y: number; w: number; h: number }; fillColor?: string }[]
}

/**
 * Inpaint all bubble regions in an image.
 * Speech bubbles → tight text rect painted white; full bubble interior returned as expandedRects.
 * Background text → LaMa ONNX reconstruction.
 *
 * @param bubbles     The detected bubbles (rect coordinates are percentage-based)
 * @param imageBlob   The raw image file
 * @param onProgress  Optional callback with (stage label, current, total)
 * @returns           InpaintResult: transparent PNG overlay blob + expanded rects per speech bubble
 */
export function inpaintPage(
  bubbles: MangaBubble[],
  imageBlob: Blob,
  onProgress?: (stage: string, current: number, total: number) => void,
  textMask?: DetectionMask | null,
): Promise<InpaintResult> {
  return new Promise((resolve, reject) => {
    const worker = getWorker()

    const handler = (e: MessageEvent) => {
      const msg = e.data
      if (msg.type === 'progress') {
        onProgress?.(msg.stage as string, msg.current as number, msg.total as number)
      } else if (msg.type === 'debug') {
        fetch('/__debug/inpaint', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(msg.data) }).catch(() => {})
      } else if (msg.type === 'done') {
        worker.removeEventListener('message', handler)
        resolve({
          blob: msg.resultBlob as Blob,
          expandedRects: msg.expandedRects as InpaintResult['expandedRects'],
        })
      } else if (msg.type === 'error') {
        worker.removeEventListener('message', handler)
        reject(new Error(msg.message as string))
      }
    }

    worker.addEventListener('message', handler)
    const msg: Record<string, unknown> = {
      type: 'inpaint',
      imageBlob,
      bubbles: bubbles.map(b => ({ id: b.id, rect: b.rect, shape: b.shape, inpaint_color: b.inpaint_color, is_background: b.is_background, rotation: b.rotation })),
    }
    if (textMask) {
      // Copy before transferring — transfer detaches the buffer, which would
      // prevent re-running inpaint (e.g. after reverting a bubble)
      const maskCopy = textMask.data.slice()
      msg.textMask = maskCopy
      msg.textMaskW = textMask.w
      msg.textMaskH = textMask.h
      worker.postMessage(msg, { transfer: [maskCopy.buffer] })
    } else {
      worker.postMessage(msg)
    }
  })
}
