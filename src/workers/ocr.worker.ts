/**
 * ocr.worker.ts
 *
 * Runs manga-ocr (kha-white/manga-ocr-base) entirely inside a Web Worker.
 * Model source: https://huggingface.co/l0wgear/manga-ocr-2025-onnx
 * Architecture: ViT encoder + BERT-based autoregressive decoder
 *
 * Message protocol:
 *   IN  { type: 'ocr', imageBlob: Blob, rect: { x, y, w, h } }
 *   OUT { type: 'progress', stage: string, value: number }
 *   OUT { type: 'result',   text: string }
 *   OUT { type: 'error',    message: string }
 *
 * Rect coordinates are percentage-based (0–100) matching MangaBubble.rect.
 */

import * as ort from 'onnxruntime-web'

// ── Config ────────────────────────────────────────────────────────────────────

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.2/dist/'

const BASE_URL   = 'https://huggingface.co/l0wgear/manga-ocr-2025-onnx/resolve/main/'
// Vocab from the original model repo — authoritative token↔ID mapping
const VOCAB_URL  = 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/vocab.txt'
const ENC_URL    = BASE_URL + 'encoder_model.onnx'
const DEC_URL    = BASE_URL + 'decoder_model.onnx'

const INPUT_SIZE  = 224
const MAX_TOKENS  = 300
const BOS         = 2n   // [CLS] token — decoder_start_token_id
const EOS         = 3n   // [SEP] token — eos_token_id

// ── Singleton models ──────────────────────────────────────────────────────────

let encoderSession: ort.InferenceSession | null = null
let decoderSession: ort.InferenceSession | null = null
let vocab: string[] | null = null

function post(msg: object): void {
  self.postMessage(msg)
}

function progress(stage: string, value: number): void {
  post({ type: 'progress', stage, value })
}

async function fetchWithProgress(
  url: string,
  label: string,
  startP: number,
  endP: number,
): Promise<ArrayBuffer> {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Fetch failed: ${resp.status} ${resp.statusText} — ${url}`)

  const total    = Number(resp.headers.get('content-length') ?? 0)
  const reader   = resp.body!.getReader()
  const chunks: Uint8Array[] = []
  let received   = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    if (total > 0) progress(label, startP + (endP - startP) * (received / total))
  }

  const buffer = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) { buffer.set(chunk, offset); offset += chunk.length }
  return buffer.buffer
}

async function getSessions(): Promise<{
  enc: ort.InferenceSession
  dec: ort.InferenceSession
  vocab: string[]
}> {
  if (encoderSession && decoderSession && vocab) {
    return { enc: encoderSession, dec: decoderSession, vocab }
  }

  // 1 — Vocab (~25 kB, instant)
  progress('Loading vocabulary…', 0.01)
  const vocabText = await fetch(VOCAB_URL).then(r => {
    if (!r.ok) throw new Error(`Vocab fetch failed: ${r.status}`)
    return r.text()
  })
  vocab = vocabText.split('\n').map(s => s.trimEnd())

  // 2 — Encoder (~22 MB)
  const encBuf = await fetchWithProgress(
    ENC_URL, 'Downloading OCR encoder (22 MB)…', 0.03, 0.20,
  )
  progress('Loading OCR encoder…', 0.20)
  encoderSession = await ort.InferenceSession.create(encBuf, { executionProviders: ['wasm'] })

  // 3 — Decoder (~118 MB)
  const decBuf = await fetchWithProgress(
    DEC_URL, 'Downloading OCR decoder (118 MB)…', 0.22, 0.88,
  )
  progress('Loading OCR decoder…', 0.88)
  decoderSession = await ort.InferenceSession.create(decBuf, { executionProviders: ['wasm'] })

  return { enc: encoderSession, dec: decoderSession, vocab }
}

// ── Preprocessing ─────────────────────────────────────────────────────────────

/**
 * Crop a bubble from the source image, resize to 224×224, convert to grayscale,
 * and normalize to float32 CHW in range [-1, 1].
 */
async function preprocess(
  blob: Blob,
  rect: { x: number; y: number; w: number; h: number },
): Promise<ort.Tensor> {
  const bitmap = await createImageBitmap(blob)
  const { width: origW, height: origH } = bitmap

  // Percentage rect → pixel rect
  const cropX = Math.round((rect.x / 100) * origW)
  const cropY = Math.round((rect.y / 100) * origH)
  const cropW = Math.max(1, Math.round((rect.w / 100) * origW))
  const cropH = Math.max(1, Math.round((rect.h / 100) * origH))

  const canvas = new OffscreenCanvas(INPUT_SIZE, INPUT_SIZE)
  const ctx    = canvas.getContext('2d')!
  ctx.drawImage(bitmap, cropX, cropY, cropW, cropH, 0, 0, INPUT_SIZE, INPUT_SIZE)
  bitmap.close()

  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
  const N       = INPUT_SIZE * INPUT_SIZE
  const floats  = new Float32Array(3 * N)

  // RGBA → luminance → 3-channel, normalize: (px/255 - 0.5) / 0.5 = px/127.5 - 1
  for (let i = 0; i < N; i++) {
    const r    = data[i * 4]
    const g    = data[i * 4 + 1]
    const b    = data[i * 4 + 2]
    const gray = 0.299 * r + 0.587 * g + 0.114 * b
    const norm = gray / 127.5 - 1.0
    floats[i]           = norm  // R channel
    floats[N + i]       = norm  // G channel
    floats[2 * N + i]   = norm  // B channel
  }

  return new ort.Tensor('float32', floats, [1, 3, INPUT_SIZE, INPUT_SIZE])
}

// ── Greedy decoding ───────────────────────────────────────────────────────────

async function greedyDecode(
  enc: ort.InferenceSession,
  dec: ort.InferenceSession,
  pixelTensor: ort.Tensor,
  vocabList: string[],
): Promise<string> {
  // Run encoder once
  const encOut       = await enc.run({ pixel_values: pixelTensor })
  const hiddenState  = encOut['last_hidden_state']   // [1, src_len, hidden]
  const encSeqLen    = hiddenState.dims[1]

  // Build encoder attention mask (all-ones) — required by some ONNX exports
  const needsEncMask = dec.inputNames.includes('encoder_attention_mask')
  const encMask      = needsEncMask
    ? new ort.Tensor('int64', BigInt64Array.from({ length: encSeqLen }, () => 1n), [1, encSeqLen])
    : null

  // Autoregressive decode: start with [BOS]
  const ids: bigint[] = [BOS]

  for (let step = 0; step < MAX_TOKENS; step++) {
    const seqLen = ids.length
    const idsTensor  = new ort.Tensor('int64',  BigInt64Array.from(ids),                   [1, seqLen])
    const maskTensor = new ort.Tensor('int64',  BigInt64Array.from({ length: seqLen }, () => 1n), [1, seqLen])

    const feed: Record<string, ort.Tensor> = {
      input_ids:             idsTensor,
      encoder_hidden_states: hiddenState,
    }
    // Only pass optional inputs if the model actually declares them
    if (dec.inputNames.includes('attention_mask'))        feed['attention_mask']        = maskTensor
    if (dec.inputNames.includes('encoder_attention_mask')) feed['encoder_attention_mask'] = encMask!

    const decOut     = await dec.run(feed)
    const logTensor  = decOut['logits']
    const logits     = logTensor.data as Float32Array
    // Use actual model vocab size from tensor dims — vocabList.length may differ by trailing newline
    const modelVocab = logTensor.dims[2]
    // logits: [1, seqLen, modelVocab] — take last token position
    const base       = (seqLen - 1) * modelVocab
    let maxVal = -Infinity, nextId = 0
    for (let v = 0; v < modelVocab; v++) {
      if (logits[base + v] > maxVal) { maxVal = logits[base + v]; nextId = v }
    }

    if (BigInt(nextId) === EOS) break
    ids.push(BigInt(nextId))
  }

  // Detokenize: skip BOS and filter special/unused tokens (IDs 0–14)
  return ids
    .slice(1)
    .filter(id => id > 14n)
    .map(id => vocabList[Number(id)] ?? '')
    .join('')
}

// ── Worker message handler ────────────────────────────────────────────────────

self.onmessage = async (e: MessageEvent) => {
  if (e.data?.type !== 'ocr') return

  const { imageBlob, rect } = e.data as {
    imageBlob: Blob
    rect: { x: number; y: number; w: number; h: number }
  }

  try {
    const { enc, dec, vocab: vocabList } = await getSessions()

    progress('Preprocessing bubble…', 0.90)
    const pixelTensor = await preprocess(imageBlob, rect)

    progress('Running OCR…', 0.92)
    const text = await greedyDecode(enc, dec, pixelTensor, vocabList)

    post({ type: 'result', text })
  } catch (err) {
    post({ type: 'error', message: String(err) })
  }
}
