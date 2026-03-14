/**
 * ocr.ts — OCR pipeline stage
 *
 * Manages the ocr Web Worker lifecycle and exposes a clean async API.
 * Heavy work (ViT encoder + BERT decoder inference) runs in ocr.worker.ts.
 *
 * Signature must not change — workspace.ts depends on it.
 */

import type { MangaBubble } from '../types'

// Singleton worker — created once, reused across OCR calls
let _worker: Worker | null = null

function getWorker(): Worker {
  if (!_worker) {
    _worker = new Worker(
      new URL('../workers/ocr.worker.ts', import.meta.url),
      { type: 'module' },
    )
  }
  return _worker
}

/**
 * Run OCR on a single bubble crop.
 * Passes the full image blob + percentage rect so the worker can crop it.
 *
 * @param bubble     The bubble to OCR (rect used for crop)
 * @param imageBlob  The full page image
 * @param onProgress Optional progress callback (stage label, 0–1 value)
 * @returns          The recognised Japanese text (raw_ja)
 */
export function runOCR(
  bubble: MangaBubble,
  imageBlob: Blob,
  onProgress?: (stage: string, value: number) => void,
): Promise<string> {
  return new Promise((resolve, reject) => {
    const worker = getWorker()

    const handler = (e: MessageEvent) => {
      const msg = e.data
      if (msg.type === 'progress') {
        onProgress?.(msg.stage, msg.value)
      } else if (msg.type === 'result') {
        worker.removeEventListener('message', handler)
        resolve(msg.text as string)
      } else if (msg.type === 'error') {
        worker.removeEventListener('message', handler)
        reject(new Error(msg.message as string))
      }
    }

    worker.addEventListener('message', handler)
    worker.postMessage({ type: 'ocr', imageBlob, rect: bubble.rect })
  })
}
