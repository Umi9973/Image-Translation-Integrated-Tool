# Web Workers Module

Workers exist to offload heavy ML tasks off the main thread so the UI never freezes.

## Files
| File | Wraps | Triggered by |
|---|---|---|
| `detect.worker.ts` | comic-text-detector ONNX (HuggingFace) | `pipeline/detect.ts` calls it on every detection run |
| `ocr.worker.ts` | manga-ocr ONNX (l0wgear/manga-ocr-2025-onnx) | `pipeline/ocr.ts` calls it per bubble; workspace "OCR All" triggers all |
| `inpaint.worker.ts` | `pipeline/inpaint.ts` | User clicks "Clean" on a page |

## detect.worker.ts — Model Details
- Model: `mayocream/comic-text-detector-onnx` on HuggingFace (~90 MB, cached after first load)
- ONNX outputs: `blk` (YOLO text blocks), `mask` (1024×1024 text-pixel probability map)
- We consume `blk` for bounding boxes and `mask` for seam detection and tightening.
- Input: 1024×1024 RGB float32, letterboxed with gray padding (114,114,114)
- Output coords: percentage-based (0–100) to match `MangaBubble.rect`
- WASM runtime files loaded from jsDelivr CDN (`onnxruntime-web@1.24.2`)

### Post-processing pipeline
1. **NMS** on raw YOLO boxes (IOU_THRESH=0.3, CONF_THRESH=0.45)
2. **`findSeamY` / `findSeamX`**: per-column/row gap analysis on the mask to detect double bubbles; dominance + vote filters; minimum box dimension guard (≥80px)
3. **`tightenToMask`**: shrink each half-box (or whole box) to the tight bounding rect of mask text-pixels — this is what gets stored in `MangaBubble.rect`
4. **False-positive split rejection**: after tightening, if the gap between the two text clusters is < `SEAM_GAP_FRAC` (20%) of total box dimension AND the perpendicular-axis overlap between the two text clusters is ≥ `SEAM_OVERLAP_FRAC` (80%) → reject the split, treat as single bubble
   - seamY (horizontal cut): checks X-overlap ratio between t1 and t2
   - seamX (vertical cut): checks Y-overlap ratio between t1 and t2
5. **`deduplicateBubbles`**: three-pass —
   - Pass 1 (wrapper): drop large box if 2+ smaller boxes are each ≥70% contained within it (spurious YOLO wrapper over individually-detected bubbles)
   - Pass 2 (containment): drop small bubble if ≥85% of its area is inside a single larger one
   - Pass 3 (coverage): drop bubble whose area is ≥85% covered by others combined (removes redundant full-box when split halves also detected independently)

### Output: tight text rects
`MangaBubble.rect` stores the **tight text-pixel bounding box**, not the full bubble interior.
Bubble border expansion (`scanBubbleBounds`) has been moved to `inpaint.worker.ts` and runs at inpaint time.

### Stable base
`detect.worker.ts.bak` — last known-good snapshot. Restore with:
```bash
cp src/workers/detect.worker.ts.bak src/workers/detect.worker.ts
```
**Update `.bak` after every confirmed-working milestone** — ask Claude to do it explicitly.

## ocr.worker.ts — Model Details
- Models: `l0wgear/manga-ocr-2025-onnx` on HuggingFace (encoder ~22 MB, decoder ~118 MB, cached after first load)
- Architecture: ViT encoder (`encoder_model.onnx`) + BERT-based autoregressive decoder (`decoder_model.onnx`)
- Vocab: `vocab.txt` (~6 k tokens, character-level Japanese), fetched from same repo
- BOS token: 2 (`[CLS]`), EOS token: 3 (`[SEP]`), PAD: 0 — defined in `generation_config.json`
- Image input: crop bubble rect from original image → resize 224×224 → grayscale → normalize (px/127.5 − 1)
- Decoding: greedy (argmax), max 300 tokens; tokens with ID ≤ 14 are special/unused and filtered out
- Encoder in: `pixel_values` [1,3,224,224] float32 → out: `last_hidden_state`
- Decoder in: `input_ids` [1,L] int64, `encoder_hidden_states` [1,src,hid] float32 → out: `logits` [1,L,vocab]
- `attention_mask` and `encoder_attention_mask` are checked via `dec.inputNames` and only passed if the model declares them (this export does NOT use them)

## inpaint.worker.ts — Hybrid Inpainting
Model: `Carve/LaMa-ONNX/lama_fp32.onnx` (~208 MB). Fetched once, saved to OPFS (`lama_fp32.onnx`), reused on subsequent runs. Only loaded when at least one bubble routes to LaMa.

### Message protocol
- **IN**: `{ type: 'inpaint', imageBlob: Blob, bubbles: Array<{id, rect: {x,y,w,h}}> }` — rect values percentage-based
- **OUT done**: `{ type: 'done', resultBlob: Blob, expandedRects: Array<{id, rect}> }` — expandedRects only for speech bubble route, percentage-based full bubble interior

### Routing — `isBrightRegion()`
Each bubble is sampled via a 5×5 grid within `SAMPLE_PAD=20px` expansion around the tight text rect (excluding the text rect itself):
- **Bright** (≥55% of samples have lum > 200) → `'white'` — speech bubble
- **Dark / colored** → `'lama'` — background text on manga artwork

Known limitation: classification can fail for ambiguous backgrounds (deferred).

### Speech bubble path — white rect
1. Paint tight text rect (tx1,ty1 → tx2,ty2) white (255,255,255,255) in `outData`
2. `scanBubbleBounds` → full bubble interior bounds (pixel coords)
3. Convert to percentage-based rect → added to `expandedRects` in response

### `scanBubbleBounds()`
Casts `SCAN_SAMPLES=9` rays per edge outward from tight text rect, stops at luminance < `DARK_THRESH=80`. Returns pixel-coordinate expanded bounds `[bx, by, bx2, by2]`. Tunables: `DARK_THRESH=80`, `MAX_EXPAND=200`, `SCAN_SAMPLES=9`.

### Background text path — LaMa
1. Bounds = tight text rect + `BG_PADDING=8px`
2. Context crop: bounds expanded by `LAMA_CTX_FRAC=0.5 × max(w,h)` per side so LaMa sees surrounding background
3. Rectangular mask covers only the bounds region within the crop
4. Scale image crop + mask → 512×512
5. `runLama()`: image `[1,3,512,512]` float32 (0–1) + mask `[1,1,512,512]` float32 → output `[1,3,512,512]`. Tensor names detected dynamically via `sess.inputNames`.
6. Scale output back to crop size
7. Paste only masked pixels (bounds region) into `outData` — context pixels are discarded

Output: transparent PNG overlay — white tight rects where speech bubbles were, LaMa-reconstructed pixels where background text was.

### Stable base
`inpaint.worker.ts.bak` — last known-good snapshot. Restore with:
```bash
cp src/workers/inpaint.worker.ts.bak src/workers/inpaint.worker.ts
```
**Update `.bak` after every confirmed-working milestone** — ask Claude to do it explicitly.

## Rules
- Workers communicate via `postMessage` / `onmessage` only — no shared state.
- Message shape must be typed. Define `WorkerRequest` and `WorkerResponse` interfaces at the top of each worker file.
- Workers may import from `../pipeline/` and `../types/` only.
- Workers must NEVER import from `../ui/` or `../db/`.
- Use `self.postMessage({ type: 'progress', value: 0.5 })` for progress updates so the UI can show a progress bar.
- Always handle errors and post `{ type: 'error', message }` back — never let a worker crash silently.

## Vite Worker Syntax
```typescript
// Instantiate in UI code like this:
const worker = new Worker(new URL('./ocr.worker.ts', import.meta.url), { type: 'module' })
```
This is required for Vite to correctly bundle the worker as a separate chunk.
