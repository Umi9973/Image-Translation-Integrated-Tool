# Pipeline Module

Each stage is a single file with a single exported function (async for pipeline stages that do I/O or ML inference, sync for pure DOM operations like typesetting).
Stages are **decoupled** — they do not import each other.
The UI orchestrates the order; pipeline files just do one job.

## Stages & Files
| File | Function export | Input | Output |
|---|---|---|---|
| `detect.ts` | `detectBubbles(imageBlob)` | `Blob` | `MangaBubble[]` (state: `detected`) |
| `ocr.ts` | `runOCR(bubble, imageBlob)` | `MangaBubble`, `Blob` | `raw_ja: string` |
| `translate.ts` | `translatePage(bubbles, providerId, apiKey, onProgress?)` | `MangaBubble[]`, provider id + key strings | `{ id, translated_zh }[]` |
| `inpaint.ts` | `inpaintPage(bubbles, imageBlob, onProgress?)` | `MangaBubble[]`, `Blob` | `InpaintResult` `{ blob: Blob, expandedRects: {id, rect}[] }` |
| `typeset.ts` | `renderTypeset(bubbles, svg)` | `MangaBubble[]`, `SVGSVGElement` | `void` (clears + redraws SVG layer) |

## Implementation Status
| File | Status |
|---|---|
| `detect.ts` | **Live** — manages detect.worker.ts singleton; exports `detectBubbles(blob, onProgress?)`. Signature must not change. Worker now runs `scanBubbleBounds()` pixel scan post-YOLO to expand each tight text box to the enclosing bubble interior. |
| `ocr.ts` | **Live** — manages ocr.worker.ts singleton; exports `runOCR(bubble, imageBlob, onProgress?)`. Signature must not change. |
| `translate.ts` | **Live** — exports `buildPrompt`, `translatePage`, `parseTranslationResponse`, `loadAPIConfig`, `saveAPIConfig`, `clearAPIConfig`. Signature must not change. Prompt instructs model to insert `\\` at natural phrase breaks. `parseTranslationResponse` includes `fixBackslashes()` sanitizer for raw `\` in model output. |
| `inpaint.ts` | **Live** — manages inpaint.worker.ts singleton; exports `inpaintPage(bubbles, imageBlob, onProgress?)` → `InpaintResult { blob, expandedRects }`. Returns a **transparent PNG overlay** + per-speech-bubble expanded interior rects. UI writes `expandedRects` into `bubble.bubble_rect` after inpainting. |
| `typeset.ts` | **Live** — no worker needed; exports `renderTypeset(bubbles, svg)`. Signature must not change. Text rendered vertically (`writing-mode="vertical-rl"`), right-to-left columns. Uses `bubble.bubble_rect ?? bubble.rect` as layout box. `\` in `translated_zh` forces a column-group break. Font: ZCOOL KuaiLe. `MAX_FONT=72`, `PADDING=6` — text fills full bubble interior as large as possible. |

## Rules
- Never import from `../ui/` — pipeline is UI-agnostic.
- OCR and Inpainting must run in a Web Worker (see `src/workers/`), not on the main thread.
- Inpainting returns `InpaintResult { blob, expandedRects }`. `blob` is a transparent PNG overlay (alpha=0 everywhere except processed regions). Speech bubble tight text rects → white. Background text regions → LaMa-reconstructed pixels. UI stamps `blob` onto `.ws-inpaint-layer` canvas and writes `expandedRects` into `bubble.bubble_rect` — original image Blob never modified.
- Routing uses `isBrightRegion()` (5×5 brightness sample): bright interior → paint white + scan full bubble bounds (`scanBubbleBounds`), dark/colored → LaMa. Known limitation: classification can fail for ambiguous backgrounds (deferred).
- `scanBubbleBounds()` in inpaint.worker.ts: casts 9 rays per edge outward, stops at luminance < 80 (dark border). Returns full bubble interior bounds (pixel coords). Tunables: `DARK_THRESH=80`, `MAX_EXPAND=200`, `SCAN_SAMPLES=9`.
- Typesetting uses `bubble.bubble_rect` (full bubble interior, set after inpainting) if available, else `bubble.rect` (tight text bbox). `\` in `translated_zh` forces a column-group boundary. Font: ZCOOL KuaiLe. `MAX_FONT=72` — text scales up to fill the bubble; `fitVertical()` descends from 72 until columns fit.
- Detection uses onnxruntime-web (WASM backend); models fetched from HuggingFace on first use.
- All bubble `rect` coordinates are **percentage-based** (0–100), not pixels.
- After each stage, update `bubble.state` to the next value in the state machine:
  `detected` → `ocr_done` → `translated` → `reviewed`
