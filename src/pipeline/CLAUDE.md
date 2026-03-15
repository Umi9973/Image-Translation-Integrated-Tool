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
| `typeset.ts` | `renderTypeset(bubbles, svg)` | `MangaBubble[]`, `SVGSVGElement` | `string[]` clipped bubble IDs |
| `typeset.ts` | `renderTypesetToCanvas(bubbles, ctx, W, H)` | `MangaBubble[]`, `CanvasRenderingContext2D`, `number`, `number` | `void` — draws text+dots onto canvas for download |

## Implementation Status
| File | Status |
|---|---|
| `detect.ts` | **Live** — manages detect.worker.ts singleton; exports `detectBubbles(blob, onProgress?)`. Signature must not change. Worker now runs `scanBubbleBounds()` pixel scan post-YOLO to expand each tight text box to the enclosing bubble interior. |
| `ocr.ts` | **Live** — manages ocr.worker.ts singleton; exports `runOCR(bubble, imageBlob, onProgress?)`. Signature must not change. |
| `translate.ts` | **Live** — exports `buildPrompt`, `translatePage`, `parseTranslationResponse`, `loadAPIConfig`, `saveAPIConfig`, `clearAPIConfig`. Signature must not change. Prompt instructs model to insert `\\` at natural phrase breaks. `parseTranslationResponse` includes `fixBackslashes()` sanitizer for raw `\` in model output. |
| `inpaint.ts` | **Live** — manages inpaint.worker.ts singleton; exports `inpaintPage(bubbles, imageBlob, onProgress?)` → `InpaintResult { blob, expandedRects }`. Returns a **transparent PNG overlay** + per-speech-bubble expanded interior rects. UI writes `expandedRects` into `bubble.bubble_rect` after inpainting. |
| `typeset.ts` | **Live** — no worker needed. Exports `renderTypeset(bubbles, svg): string[]` (returns clipped-dot bubble IDs) and `renderTypesetToCanvas(bubbles, ctx, W, H): void` (used for download). Text is vertical right-to-left columns, font ZCOOL KuaiLe. Config constants: `MAX_FONT=72`, `MIN_FONT=8`, `PADDING=6`, `COL_GAP=4`, `DOT_RADIUS=2.2`, `DOT_STRIDE=9`, `DOT_OVERFLOW_FACTOR=1.5`. Pre-rendering pipeline per bubble: `normalizeVertical` → BudouX → `mergetitles` → `mergeparticles` → `mergedots` → `fitVertical` → `packChunks`. Dots (`・`) are drawn as geometric circles (SVG `<circle>` / canvas `arc`) with fixed radius and stride, never font glyphs. Dot runs are never hard-split across columns; excess dots clipped when they exceed `innerH × DOT_OVERFLOW_FACTOR / DOT_STRIDE`. `\` in `translated_zh` forces a column-group break. |

## Rules
- Never import from `../ui/` — pipeline is UI-agnostic.
- OCR and Inpainting must run in a Web Worker (see `src/workers/`), not on the main thread.
- Inpainting returns `InpaintResult { blob, expandedRects }`. `blob` is a transparent PNG overlay (alpha=0 everywhere except processed regions). Speech bubble tight text rects → white. Background text regions → LaMa-reconstructed pixels. UI stamps `blob` onto `.ws-inpaint-layer` canvas and writes `expandedRects` into `bubble.bubble_rect` — original image Blob never modified.
- Routing uses `isBrightRegion()` (5×5 brightness sample): bright interior → paint white + scan full bubble bounds (`scanBubbleBounds`), dark/colored → LaMa. Known limitation: classification can fail for ambiguous backgrounds (deferred).
- `scanBubbleBounds()` in inpaint.worker.ts: casts 9 rays per edge outward, stops at luminance < 80 (dark border). Returns full bubble interior bounds (pixel coords). Tunables: `DARK_THRESH=80`, `MAX_EXPAND=200`, `SCAN_SAMPLES=9`.
- Typesetting uses `bubble.bubble_rect` (full bubble interior, set after inpainting) if available, else `bubble.rect` (tight text bbox). `\` in `translated_zh` forces a column-group boundary. Font: ZCOOL KuaiLe. `MAX_FONT=72` — text scales up to fill the bubble; `fitVertical()` descends from 72 until columns fit.
- ASCII punctuation and Unicode ellipsis (`…`) are normalised to CJK full-width equivalents before BudouX (`normalizeVertical`). Dot sequences map to `・` runs. Dashes (`－`, `—`) are rotated 90° in the canvas renderer since canvas has no `writing-mode`.
- `mergetitles` keeps name+title (e.g. 奈良先生) in one column. `mergeparticles` keeps sentence-final particles (吗呢吧啊嘛哦哈呀 etc.) attached to the preceding chunk so they never open a column alone. `mergedots` keeps dot runs unsplit and attached to adjacent word chunks so they land in the same column. `packChunks` hard-split: when splitting a long chunk at `splitAt = charsPerCol`, if `rem[splitAt]` is a particle, bumps `splitAt++` to keep it with the preceding characters — prevents `是吗` from being separated when the split boundary falls exactly on `吗`.
- Detection uses onnxruntime-web (WASM backend); models fetched from HuggingFace on first use.
- All bubble `rect` coordinates are **percentage-based** (0–100), not pixels.
- After each stage, update `bubble.state` to the next value in the state machine:
  `detected` → `ocr_done` → `translated` → `reviewed`
