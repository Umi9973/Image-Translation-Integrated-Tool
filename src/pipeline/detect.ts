/**
 * detect.ts — Detection pipeline stage
 *
 * Manages the detect Web Worker lifecycle and exposes a clean async API.
 * Heavy work (ONNX inference, NMS) runs in detect.worker.ts, not the main thread.
 *
 * Signature must not change — workspace.ts depends on it.
 */

import type { MangaBubble } from '../types'

// Singleton worker — created once, reused across detections
let _worker: Worker | null = null

function getWorker(): Worker {
  if (!_worker) {
    _worker = new Worker(
      new URL('../workers/detect.worker.ts', import.meta.url),
      { type: 'module' },
    )
  }
  return _worker
}

/**
 * Detect all text regions (speech bubbles + background text) in an image.
 * Uses the comic-text-detector ONNX model via a Web Worker.
 *
 * @param imageBlob  The raw image file
 * @param onProgress Optional callback for loading/inference progress (stage label, 0–1 value)
 */
export function detectBubbles(
  imageBlob: Blob,
  onProgress?: (stage: string, value: number) => void,
): Promise<MangaBubble[]> {
  return new Promise((resolve, reject) => {
    const worker = getWorker()

    const handler = (e: MessageEvent) => {
      const msg = e.data
      if (msg.type === 'progress') {
        onProgress?.(msg.stage, msg.value)
      } else if (msg.type === 'debug') {
        fetch('/__debug', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(msg.data) }).catch(() => {})
      } else if (msg.type === 'result') {
        worker.removeEventListener('message', handler)
        resolve(msg.bubbles as MangaBubble[])
      } else if (msg.type === 'error') {
        worker.removeEventListener('message', handler)
        reject(new Error(msg.message as string))
      }
    }

    worker.addEventListener('message', handler)
    worker.postMessage({ type: 'detect', imageBlob })
  })
}
