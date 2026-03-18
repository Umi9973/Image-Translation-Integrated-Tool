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
2. **`findSeamY` / `findSeamX`**: gap analysis on the mask to detect double bubbles; minimum box dimension guards: seamY requires both height AND width ≥ 80px (narrow boxes can't be stacked double-bubbles); seamX requires width ≥ 80px
   - **`findSeamX`** (left/right split): per-row vote — counts how many rows have a dark column gap; needs `MIN_VOTE_FRAC=0.20` of rows to agree. Works because vertical text columns span the full height so every row sees both sides.
   - **`findSeamY`** (top/bottom split): horizontal projection profile — for each row counts text-pixel columns; finds actual text extent (trims YOLO box padding); computes quarter averages anchored to text extent; finds the **widest contiguous near-zero band** in the middle 40–80% of the text extent where per-row density ≤ `SEAM_Y_VALLEY_ABS_FRAC` (3% of box width) AND band is ≥ `SEAM_Y_MIN_BAND_FRAC` (5% of text height) rows AND band average ≤ `SEAM_Y_VALLEY_FRAC` (15%) of the weaker half's average. The widest-band requirement distinguishes real inter-bubble gaps (10+ rows) from DBNet inter-character gaps (1–2 rows). Per-column voting fails for stacked bubbles because both bubbles share the same X positions — no column sees text from two different vertical positions.
3. **`tightenToMask`**: shrink each half-box (or whole box) to the tight bounding rect of mask text-pixels — this is what gets stored in `MangaBubble.rect`
4. **False-positive split rejection**: after tightening:
   - seamX: if gap between text clusters < `SEAM_GAP_FRAC` (20%) of box width AND Y-overlap ≥ `SEAM_OVERLAP_FRAC` (80%) → reject split
   - seamY: revert if `gapH < SEAM_Y_MIN_GAP_FRAC × boxHeight` (i.e. tight-rect gap < 4% of YOLO box height). Real inter-bubble gaps are ≥ 4% of box height; DBNet inter-character spacing (3–8 px) is not. Do NOT use X-overlap as discriminator — stacked bubbles always share the same horizontal span, so X-overlap would always fire and reject real splits.
5. **`deduplicateBubbles`**: three-pass —
   - Pass 1 (wrapper): drop large box if 2+ smaller boxes are each ≥70% contained within it (spurious YOLO wrapper over individually-detected bubbles)
   - Pass 2 (containment): drop small bubble if ≥85% of its area is inside a single larger one
   - Pass 3 (coverage): drop bubble whose area is ≥85% covered by others combined (removes redundant full-box when split halves also detected independently)

### Output: tight text rects
`MangaBubble.rect` stores the **tight text-pixel bounding box**, not the full bubble interior.
Bubble border expansion (`scanBubbleBounds`) has been moved to `inpaint.worker.ts` and runs at inpaint time.

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

### Routing — three-way classification
1. **`isBrightRegion()`** — samples a 5×5 grid **inside** the tight text rect (not around it, to avoid being fooled by dark panel borders):
   - ≥50% samples have luminance > 200 → **`'white'`** (speech bubble, white interior)
   - Otherwise → check `sampleBorderColor()`
2. **`sampleBorderColor()`** — samples a `SOLID_RING=12px` wide ring just outside the text rect at 4px intervals, computes per-channel mean + stddev:
   - max channel stddev < `SOLID_THRESH=28` → **`'solid'`** (uniform panel background)
   - Otherwise → **`'lama'`** (complex manga artwork background)

### Speech bubble path — `'white'`
1. `scanBubbleBounds` → full bubble interior bounds (pixel coords)
2. Paint expanded tight rect (+ `WHITE_EXPAND=7px` where room allows) white in `outData`
3. Convert to percentage-based rect → added to `expandedRects` in response

### `scanBubbleBounds()`
Casts `SCAN_SAMPLES=9` rays per edge outward from tight text rect, stops at luminance < `DARK_THRESH=80`. Returns pixel-coordinate expanded bounds `[bx, by, bx2, by2]`. Tunables: `DARK_THRESH=80`, `MAX_EXPAND=200`, `SCAN_SAMPLES=9`.

### Solid background path — `'solid'`
1. Uses the already-computed `sampleBorderColor()` mean RGB
2. Fills tight rect + `BG_PADDING=8px` with that averaged color
3. Fast, no model needed — handles text on colored/toned panels

### Background text path — `'lama'`
1. Bounds = tight text rect + `BG_PADDING=8px`
2. Context crop: bounds expanded by `LAMA_CTX_FRAC=0.5 × max(w,h)` per side so LaMa sees surrounding background
3. Pixel-level mask: within bounds, mask only pixels with luminance > `TEXT_LUM_THRESH=160` (text ink on dark background); fallback to full rect if < 1% pixels masked
4. Scale image crop + mask → 512×512
5. `runLama()`: image `[1,3,512,512]` float32 (0–1) + mask `[1,1,512,512]` float32 → output `[1,3,512,512]`. Tensor names detected dynamically via `sess.inputNames`.
6. Scale output back to crop size; paste only masked pixels into `outData`
7. OPFS cache auto-recovers: if `InferenceSession.create` throws (corrupted cache), deletes the cached file and re-downloads before retrying

Output: transparent PNG overlay — white rects (speech bubbles), solid-color fills (uniform-bg text), LaMa-reconstructed pixels (artwork-bg text).

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
