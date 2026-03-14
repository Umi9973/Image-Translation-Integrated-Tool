/**
 * inpaint.ts — Inpainting pipeline stage
 *
 * Manages the inpaint Web Worker lifecycle and exposes a clean async API.
 * Heavy work (LaMa ONNX inference) runs in inpaint.worker.ts, not the main thread.
 *
 * Signature must not change — workspace.ts depends on it.
 */

import type { MangaBubble } from '../types'

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
  expandedRects: { id: string; rect: { x: number; y: number; w: number; h: number } }[]
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
): Promise<InpaintResult> {
  return new Promise((resolve, reject) => {
    const worker = getWorker()

    const handler = (e: MessageEvent) => {
      const msg = e.data
      if (msg.type === 'progress') {
        onProgress?.(msg.stage as string, msg.current as number, msg.total as number)
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
    worker.postMessage({
      type: 'inpaint',
      imageBlob,
      bubbles: bubbles.map(b => ({ id: b.id, rect: b.rect })),
    })
  })
}
