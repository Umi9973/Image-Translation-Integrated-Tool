# Known Issues & Post-Mortems

---

## Issue #1 — OCR decoder rejected `attention_mask` input

**Status:** Fixed
**Date:** 2026-03-04
**Affected file:** `src/workers/ocr.worker.ts`

### Symptom
After clicking "OCR All", the browser console showed:
```
Error: invalid input 'attention_mask'
```
Every single bubble failed. The `raw_ja` field stayed empty for all bubbles.

### Root cause
The OCR pipeline uses two ONNX models from `l0wgear/manga-ocr-2025-onnx`:
- `encoder_model.onnx` — ViT image encoder
- `decoder_model.onnx` — BERT-based autoregressive text decoder

The decoder was called with three inputs:
```typescript
{ input_ids, attention_mask, encoder_hidden_states }
```

This particular ONNX export of the manga-ocr decoder does **not** declare `attention_mask` as an input node. ONNX Runtime Web throws a hard error if you pass any input that is not present in the model's graph, even if it would otherwise be ignored.

The original code assumed `attention_mask` was always required (it is for many other seq2seq ONNX models), but this specific export omits it because the decoder uses built-in causal masking.

### Fix — `src/workers/ocr.worker.ts` (function `greedyDecode`)
Instead of always including `attention_mask`, we now check the model's declared inputs at runtime before adding any optional tensor:
```typescript
const feed: Record<string, ort.Tensor> = {
  input_ids:             idsTensor,
  encoder_hidden_states: hiddenState,
}
if (dec.inputNames.includes('attention_mask'))         feed['attention_mask']         = maskTensor
if (dec.inputNames.includes('encoder_attention_mask')) feed['encoder_attention_mask'] = encMask!
```
`dec.inputNames` is a string array provided by ONNX Runtime that lists exactly what the loaded model accepts. This approach is safe regardless of which ONNX export variant is used in the future.

---

## Issue #2 — OCR produced garbled output due to vocabSize off-by-one

**Status:** Fixed
**Date:** 2026-03-04
**Affected file:** `src/workers/ocr.worker.ts`

### Symptom
OCR ran without errors, but every bubble produced completely wrong Japanese characters. Example:

- Expected: `ステステ`
- Actual: `ジメツゝチヲデテグツパズスジラれぎへ螳战ヰデゴ訌懺☉☆竄●鷲丹葵ィアれ゠ホゝ゚婷憩筮…`

The model always hit the 300-token limit without producing the EOS token (`[SEP]`, ID 3), meaning it looped forever generating wrong characters.

### Root cause
The logits tensor from the decoder has shape `[1, seqLen, vocabSize]` and is stored as a flat Float32Array in memory. To read the logits for the **last** token position you must compute a byte offset:
```
base = (seqLen - 1) × vocabSize
```

The original code used `vocabSize = vocabList.length`, where `vocabList` was built by:
```typescript
vocab = vocabText.split('\n').map(s => s.trimEnd())
```

`vocab.txt` ends with a newline character (`\n`), so `split('\n')` produces one trailing empty string at the end. This made:
```
vocabList.length  = 6145   ← one too many
model logits dim  = 6144   ← actual vocab size
```

With this mismatch, the base offset was wrong on every step:
```
step 0 (seqLen=1): base = 0        → correct by luck (seqLen-1 = 0)
step 1 (seqLen=2): base = 6145     → should be 6144, reads 1 float too late
step 2 (seqLen=3): base = 12290    → should be 12288, now 2 floats off
step N:            base = N×6145   → error grows linearly with sequence length
```

By step 2 the argmax was reading from the wrong slice of the logits array entirely, selecting a random high-value float that happened to be there rather than the genuine next-token prediction. The character mapped to that index was garbage, and since it was never EOS, the loop continued for all 300 steps.

This was the worst kind of bug: no exception, no visible error, just subtly wrong numbers that produced plausible-looking (but incorrect) Unicode characters.

### Fix — `src/workers/ocr.worker.ts` (function `greedyDecode`)
Read the true vocabulary dimension directly from the tensor's own shape metadata instead of deriving it from the vocab file:
```typescript
// Before (wrong):
const vocabSize = vocabList.length           // 6145 due to trailing newline

// After (correct):
const modelVocab = logTensor.dims[2]         // 6144 — always matches the actual model
const base       = (seqLen - 1) * modelVocab
```
The `vocabList` array is still used for the final ID→character mapping (`vocabList[nextId]`), but is no longer used for the critical offset arithmetic.

### Secondary fix — vocab source
As an additional hardening measure, the vocab URL was switched from the l0wgear mirror to the original `kha-white/manga-ocr-base` repository, which is the authoritative source for the token↔ID mapping:
```typescript
// Before:
const VOCAB_URL = 'https://huggingface.co/l0wgear/manga-ocr-2025-onnx/resolve/main/vocab.txt'

// After:
const VOCAB_URL = 'https://huggingface.co/kha-white/manga-ocr-base/resolve/main/vocab.txt'
```

### General lesson
When working with flat-array tensor data, **always derive stride/offset constants from the tensor's own `.dims` array**, never from an external source like a file line count. A one-item difference in an assumed dimension compounds linearly as sequence length grows.

---

## Issue #3 — Inpaint layer appeared as opaque rectangle over original image

**Status:** Fixed
**Date:** 2026-03-06
**Affected files:** `src/workers/inpaint.worker.ts`, `src/ui/workspace.ts`

### Symptom
After running "Inpaint All", the `.ws-inpaint-layer` canvas showed a solid (non-transparent) region covering the full bounding rectangle of each detected bubble. On manga pages with dark artwork surrounding the bubble, this appeared as a conspicuous opaque patch that did not match the oval bubble shape.

### Root cause (two compounding problems)

**Problem A — Full-image blob returned to UI.**
`inpaintAll()` in the worker composited each bubble's LaMa result back onto `mainCanvas`, which held the entire original image. It then exported `mainCanvas` as the result blob. The UI received a fully opaque full-page PNG and drew only the bubble rect regions from it onto the transparent `inpaintCanvas`. While logically correct, this was fragile: any coordinate or dimension mismatch would cause the wrong pixels to appear.

**Problem B — Rectangular mask and paste bleeds outside the oval bubble.**
The mask tensor sent to LaMa was a filled rectangle aligned to the bubble bounding box. Speech bubbles are oval/elliptical — the corners of the rectangle extend outside the bubble border into the surrounding artwork. LaMa inpainted those corner pixels too, and the rectangular paste-back wrote them onto the canvas, producing visible corner patches outside the bubble edge.

### Fix

**Worker (`inpaint.worker.ts`):**
1. Added a dedicated `outCanvas` (W×H, fully transparent) alongside the existing `mainCanvas`. Only the elliptical bubble region is ever drawn to `outCanvas`.
2. Changed the mask tensor from a filled rectangle to a filled **ellipse** inscribed in the bounding box using `(dx/rx)²+(dy/ry)²≤1`. This ensures LaMa only fills the interior of the oval bubble, not the corners.
3. Both `mainCtx` and `outCtx` now use `ctx.ellipse()`+`ctx.clip()` before drawing the LaMa result, so the paste is also clipped to the ellipse even for the pixels LaMa didn't mask.
4. The returned blob is now `outCanvas.convertToBlob()` — a transparent PNG where only the elliptical bubble interiors have pixels.

**UI (`workspace.ts`):**
Replaced the per-bubble copy loop with a single `ctx.clearRect()` + `ctx.drawImage(blob, 0, 0, W, H)`. Because the blob is already transparent everywhere except bubble ellipses, a full-canvas stamp is sufficient and guaranteed correct.

### How the root cause was identified
The key insight came from comparing the problem to the manual Photoshop scanlation workflow: a scanlator selects *inside* the bubble, fills with white, then places translated text on top. The oval bubble border is never touched. Our rectangular mask was equivalent to selecting a bounding box instead of the oval interior — the corners of the selection visibly exceeded the bubble border.

### General lesson
When compositing onto a transparent layer, **generate a transparent output at the source** (worker) rather than compositing at the destination (UI). This eliminates an entire class of coordinate-mismatch bugs. For oval/irregular shapes, always clip both the AI mask and the canvas paste to that shape.

---

## Issue #4 — Double bubble (peanut/figure-8) detected as one bubble instead of two

**Status:** Fixed
**Date:** 2026-03-06
**Affected file:** `src/workers/detect.worker.ts`

### Symptom
A double speech bubble — two elliptical chambers fused into a peanut / figure-8 shape with the inner border erased — was returned by YOLO as a single bounding box. The app displayed it as one bubble with a combined OCR result, rather than two independent bubbles each containing their own text.

### Background: what a double bubble is
In manga, a double bubble is two overlapping ellipses where the internal dividing border has been erased by the artist. The result is a continuous outer border enclosing two distinct chambers. Each chamber contains its own text. YOLO sees one large rectangle; human readers see two separate bubbles.

### Approaches tried (and why they failed)

**Attempt 1 — Row-by-row mask profile.**
Summed text-pixel counts per row across the full bounding box height, looked for a row range with zero sum. Failed because individual kanji have small vertical gaps between characters that produced false zero-rows; inter-character spacing was indistinguishable from inter-chamber spacing.

**Attempt 2 — Interior pixel-width scan (`findPinchY`).**
For each row, counted non-white pixels to detect a "pinch" (narrowing) at the junction of the two chambers. Failed because the center column is not reliably the widest; curved bubble borders made the width profile noisy and any threshold produced false splits on single bubbles.

**Attempt 3 — Per-column gap scan anchored at midY.**
For each mask column, found the last text row above the box midpoint and the first text row at or after midpoint, then measured the gap between them. Produced inflated measurements because midY often landed inside a character span, so the "gap" counted non-gap pixels. Large-font single bubbles like `はい?` (three tall kanji) triggered false splits.

### Final fix — `src/workers/detect.worker.ts` (`findSeamY` + `findSeamX`)

**Core algorithm (both directions use the same logic, transposed):**

For each column (Y-seam) or row (X-seam) of the mask crop:
1. Walk the slice and collect all consecutive text runs.
2. Identify all **gaps** between consecutive runs (pairs of runs with no text between them).
3. The largest gap must pass two filters:
   - **Size filter:** `largestGap ≥ MIN_GAP_FRAC × boxDimension` (proportional, default 15%).
   - **Dominance filter:** `largestGap ≥ MIN_DOMINANCE × secondLargestGap` (default 2.5×). This rejects single bubbles where all inter-character gaps are similar size (ratio ≈ 1); double bubbles have one gap 3–6× larger than the rest.
4. The gap's centre must fall in the middle 20–80% of the box (avoids edge artefacts).
5. Slices that pass all filters cast a vote for their gap-centre coordinate. If ≥ `MIN_VOTE_FRAC` (25%) of slices vote, the median vote is the seam coordinate.

Y-seam is tried first; X-seam is tried only if no Y-seam is found (a box cannot split in both directions).

**Tight rects + deferred expansion (2026-03-12 architecture change):**
`scanBubbleBounds` was moved out of `detect.worker.ts` entirely. Detection now outputs tight text-pixel rects (`tightenToMask` result per half). `inpaint.worker.ts` runs `scanBubbleBounds` at inpaint time to recover the full bubble interior. This eliminated the need for seam clamping: since the tight text rect is well inside each sub-bubble, `scanBubbleBounds` stops at that sub-bubble's real border and never crosses into the adjacent chamber.

**Final constants:**
```typescript
const SPLIT_MIN_H      = 40    // minimum px dimension per sub-bubble
const MASK_TEXT_THRESH = 0.3   // mask value above this = text pixel
const MIN_GAP_FRAC     = 0.15  // seam gap must be ≥ 15% of box dimension
const MIN_DOMINANCE    = 2.5   // seam gap must be ≥ 2.5× second-largest gap
const MIN_VOTE_FRAC    = 0.20  // ≥ 20% of slices must agree on a gap
```

**Minimum box dimension guard:**
Before running seam detection, the box must be at least `SPLIT_MIN_H * 2 = 80px` wide (for Y-seam) or tall (for X-seam). This rejects narrow single text columns (e.g. two vertical kanji like "はい" in a 41px-wide box) that would otherwise pass all filters due to the inter-character gap looking identical to a double-bubble seam.

### Issue #4b — Tuning refinements (follow-up session 2026-03-12)

**False positive:** A two-character vertical text column "はい" (41px wide, 118px tall) was being split by Y-seam detection. Root cause: the gap between は and い in a narrow column passes the dominance and size filters because the inter-character gap is the only gap and is therefore trivially dominant. Fix: added minimum box width check (`w ≥ 80px`) before attempting Y-seam.

**False negative:** A large double bubble (278×441px) was not being split. Root cause: the two chambers' text regions had limited vertical overlap, so fewer rows than expected saw text in both chambers. The vote count reached 21.7% (48/221 rows) but the threshold was 25%. Fix: lowered `MIN_VOTE_FRAC` from 0.25 → 0.20.

### General lesson
When splitting compound detections, **measure gaps proportionally** (relative to box size) rather than using fixed pixel thresholds — this adapts to varying bubble scales. Apply a **dominance filter** (largest gap vs. second-largest) rather than an absolute threshold: it correctly rejects single bubbles where inter-character gaps are all similar, while accepting double bubbles where one gap is clearly dominant. Always **clamp the inner edges** of each split result to the seam coordinate when the internal border has been erased. Add a **minimum box dimension guard** to prevent narrow single text columns from being falsely split — a real double bubble must be wide (or tall) enough to contain two separate speech bubble interiors.

---

## Issue #5 — LaMa inpainting is slow (WASM / WebGL)

**Status:** Open (unsolved)
**Date:** 2026-03-12
**Affected file:** `src/workers/inpaint.worker.ts`

### Symptom
Inpainting a page with several bubbles takes noticeably long. Each bubble requires a full 512×512 LaMa ONNX inference pass, run sequentially.

### Investigation
- Original backend: `executionProviders: ['wasm']`
- Tried: `executionProviders: ['webgl', 'wasm']` — WebGL is attempted first with WASM as fallback
- Result: no meaningful speedup observed

### Likely causes / ideas not yet tried
- The LaMa model is fp32 (large). An fp16 or quantized variant would reduce compute significantly.
- WASM ONNX Runtime is single-threaded by default; enabling multi-threading (`ort.env.wasm.numThreads`) may help.
- WebGL EP may not be initialising correctly in the worker context (no error thrown, but EP silently falls back to WASM). Worth logging `session.handler` or timing individual runs to confirm which EP is active.
- Batching all bubbles into a single inference call (if model supports dynamic batch) would amortise model-dispatch overhead.
- Running inferences in parallel Web Workers (one per bubble) could halve wall-clock time at the cost of memory.

---

## Issue #6 — Single bubble wrongly split into two by seam detection

**Status:** Fixed (pending verification)
**Date:** 2026-03-12
**Affected file:** `src/workers/detect.worker.ts`

### Symptom
Some single-chamber speech bubbles — particularly those containing two distinct vertical text columns with a large gap between them — are incorrectly split into two separate bubbles by the seam detector. The resulting rectangles each cover only half the real bubble, causing bad inpainting and wrong OCR.

### Root cause
`findSeamY` / `findSeamX` finds the largest inter-text gap in the mask and votes on a seam. For a tall single bubble with two vertical text columns, the horizontal gap between those columns can pass the size filter (≥15% of box width), the dominance filter (only one large gap), and the vote threshold — making it indistinguishable from a real double-bubble seam by the current algorithm.

### Improvements applied (2026-03-12)
- **`tightenToMask`**: outputs tight text-pixel rects per split half (and per whole box). These are now what gets stored in `MangaBubble.rect`.
- **`deduplicateBubbles`**: three-pass system (see below).
- **Stable base file**: `detect.worker.ts.bak` created as restore point.

### Attempted fix — inward ray overlap check (reverted multiple times)
Approach: shoot rays inward from each half's tight text box toward the seam; if both rays overshoot and overlap, it's a false positive. Reverted each time because the algorithm caused complete detection failure in practice — likely because erased inner borders caused `scanBubbleBounds` to overshoot in ways that made the ray threshold unreliable.

### Final fix — tight-box coordinate analysis + perpendicular-axis overlap (2026-03-13)
After `tightenToMask` on each half, check two conditions (both must pass to revert to single bubble):
1. **Gap check**: `gapH / totalH < SEAM_GAP_FRAC (0.20)` — text clusters nearly touch across the seam
2. **Perpendicular-axis overlap**: overlap of the two tight rects on the axis perpendicular to the cut ≥ `SEAM_OVERLAP_FRAC (0.80)`:
   - seamY (horizontal cut) → X-overlap ratio between t1 and t2
   - seamX (vertical cut) → Y-overlap ratio between t1 and t2
   - High overlap means both text clusters occupy the same column/row → one bubble wrongly split

The fill-ratio check (`SEAM_FILL_FRAC`) was replaced by the perpendicular-axis overlap check, which is more geometrically meaningful: two text clusters belonging to one bubble naturally align on the perpendicular axis; two genuinely separate bubbles may sit at different positions.

### deduplicateBubbles — three-pass system (2026-03-13)
- **Pass 1 (wrapper)**: drop large box if 2+ smaller boxes are each ≥70% contained within it — catches spurious YOLO wrapper boxes around individually-detected bubbles. Runs first so smaller boxes are still alive.
- **Pass 2 (containment)**: drop small bubble if ≥85% of its area is inside a single larger one.
- **Pass 3 (coverage)**: drop bubble whose area is ≥85% covered by others combined — removes redundant full-box when split halves are also detected independently.

---

## Issue #6b — Inpaint fill was faint grey instead of white inside white bubbles

**Status:** Fixed
**Date:** 2026-03-26
**Affected file:** `src/workers/inpaint.worker.ts`

### Symptom
After inpainting a white speech bubble, the erased area appeared as a faint grey rectangle or grey text-shaped shadow rather than clean white. The grey was not from leftover text pixels but from the fill color itself being wrong.

### Root cause (three compounding problems)

**Problem A — Hardcoded grey fallback.**
`sampleBackgroundFromMask` samples a narrow band of non-text pixels near text strokes to estimate background color. If fewer than 4 samples are collected, it returns a hardcoded `{ r: 128, g: 128, b: 128 }`. For white bubbles, the fallback ring (which expands outward from the text rect) filtered out near-white pixels via `HALO_WHITE_THRESH`, leaving zero samples → hardcoded grey → solid route with grey fill.

**Problem B — HALO_WHITE_THRESH in fallback ring.**
The fallback ring intentionally filtered near-white pixels to avoid white halos contaminating background color estimates for non-white regions. But this filter also removed the actual bubble interior white pixels, guaranteeing zero samples for white bubbles and triggering the hardcoded grey fallback.

**Problem C — Ellipse fill missed text corners.**
In the white route, the fill used an ellipse inscribed in the padded text rect. Japanese text characters near the corners of the bounding box fall outside the inscribed ellipse, leaving those pixels unfilled. Since the inpaint canvas is a transparent overlay, unfilled pixels show the original grey text through.

### Fix
1. **Remove `HALO_WHITE_THRESH` filter from fallback ring** — fallback now samples all non-text pixels including white, so white bubbles correctly return near-white and route to the white path.
2. **Hardcode white route fill to `(255, 255, 255)`** — the white route is only reached after confirming the bubble is white; sampling and averaging introduces faint grey from anti-aliased border pixels. Pure white is always correct here.
3. **Replace ellipse fill with rectangle fill** — text erasure should cover the full padded text rect regardless of bubble visual shape; corners must be covered.

### General lesson
When a function has a hardcoded fallback value, **make sure the fallback is detectable as a failure** (e.g. return null/undefined) rather than a plausible-looking value like 128. A grey fill that looks "almost right" is harder to debug than a clearly wrong value.

---

## Issue #7 — False positive detections (leaves, patterns, non-text regions)

**Status:** Fixed
**Date:** 2026-03-26
**Affected file:** `src/workers/detect.worker.ts`

### Symptom
After detection, bubbles appeared over image regions that clearly contained no text — leaves, background patterns, decorative artwork. These had to be manually deleted before inpainting.

### Root cause
The detection pipeline filtered by YOLO confidence (`CONF_THRESH = 0.45`) and NMS IoU overlap, but never validated whether the detection heatmap (textMask) actually contained text pixels inside each box. The YOLO model sometimes fired on visually interesting regions that resembled text structure, even though the segmentation heatmap — a separate model output trained specifically to identify text strokes — showed nothing inside the box.

### Fix
After NMS, compute **textMask density** for each kept box: the fraction of pixels inside the model-scale box where `maskData > MASK_TEXT_THRESH (0.3)`. Discard any box where density is below a minimum threshold.

```typescript
const MIN_MASK_DENSITY = 0.05  // boxes with < 5% text pixels in the heatmap are false positives
if (maskData && maskDensity < MIN_MASK_DENSITY) return []
```

The threshold was calibrated from real detection data. All confirmed false positives had density ≈ 0 (0.000–0.003). All confirmed real text bubbles had density ≥ 0.1.

### Debug tooling added
- `det_conf` and `det_mask_density` are now stamped on each returned `MangaBubble`.
- After each detection run, `detect_sorted-debug.json` is written to the project root with entries ordered by bubble panel number, so specific bubbles can be looked up by their UI number.

### General lesson
When a pipeline has two independent models (object detector + segmentation mask), **use both signals for filtering** — neither alone is sufficient. The YOLO confidence catches low-signal regions; the mask density catches cases where YOLO was confident but the pixel-level evidence disagrees.

---

## Issue #8 — Inpaint fill was rectangular for oval speech bubbles; re-inpaint after revert reverted to rectangle

**Status:** Fixed
**Date:** 2026-03-28
**Affected file:** `src/workers/inpaint.worker.ts`, `src/ui/workspace.ts`

### Symptom
White speech bubbles showed a rectangular white fill patch after inpainting — the fill extended into the corners outside the oval bubble border. Additionally, if the user hit "Revert All Inpaint" and ran inpaint again on a bubble-shaped box, the fill was a rectangle again even though the first run had been oval.

### Root cause (two problems)

**Problem A — White route always used rectangle fill.**
The white route filled the padded text rect with a solid rectangle regardless of the bubble's `shape` field. Oval speech bubbles have corners that extend outside the bubble border, so the rectangular fill left visible white patches outside the oval edge.

**Problem B — Re-inpaint after revert forced the solid (rect) route.**
After the first inpaint, the worker returned the fill color to `workspace.ts`, which stored it as `b.inpaint_color = '#ffffff'`. On the second run, the routing logic checked `b.inpaint_color` first and short-circuited to the solid route (which uses a rectangle), bypassing the shape-aware white route entirely. The root cause was treating white as a "custom color override" rather than the default bubble fill.

### Fix
1. **Shape-aware white fill** — white route now uses an ellipse inscribed in the padded text rect when `b.shape === 'bubble'`, plain rectangle otherwise.
2. **Don't cache white as inpaint_color** — `workspace.ts` now skips storing `inpaint_color` when the returned fill is `#ffffff`:
   ```typescript
   if (fillColor && fillColor !== '#ffffff') b.inpaint_color = fillColor
   ```

### General lesson
Treat default/automatic values differently from user-set overrides. Caching the default value (white) as an explicit override made the system forget which route was appropriate on the next run.

---

## Issue #9 — Typeset button stayed disabled after manual box + inpaint + translation

**Status:** Fixed
**Date:** 2026-03-28
**Affected file:** `src/ui/workspace.ts`

### Symptom
When a user manually drew a detection box, inpainted it, then typed a translation into the editor, the Typeset button remained greyed out. The badge also did not advance to `translated` state.

### Root cause
There were two separate `onTextChange` handler paths in `workspace.ts`:
1. The `selectBubble` path — correctly called `updateTypesetBtn()` after updating `translated_zh`.
2. The OCR completion path (used after OCR finishes and the editor is already open) — updated `translated_zh` and the state badge but was missing the `updateTypesetBtn()` call.

Manually added boxes go through a flow closer to the OCR completion path (no prior `selectBubble` call that registers the first handler), so they always hit the broken second path.

### Fix
Added `updateTypesetBtn()` to the OCR completion `onTextChange` handler:
```typescript
if (field === 'translated_zh') {
  updateTypesetBtn()
  if (bubble.state === 'detected' || bubble.state === 'ocr_done') { … }
}
```

---

## Issue #10 — Background text inpaint fill improvements

**Status:** Fixed
**Date:** 2026-03-28
**Affected file:** `src/workers/inpaint.worker.ts`

### Problem A — Fill did not rotate with the box

#### Symptom
When a background text box had a `rotation` set, the solid fill remained axis-aligned while the box visually showed at an angle, leaving original pixels visible in the rotated corners.

#### Fix
The `is_background === true` fill path now applies the same rotated-AABB approach used elsewhere: compute the axis-aligned bounding box of the rotated rect, iterate over it, and skip pixels that fall outside the rotated rect via the local-frame test:
```typescript
const lx = dx * cosA - dy * sinA, ly = dx * sinA + dy * cosA
if (Math.abs(lx) > rw || Math.abs(ly) > rh) continue
```

### Problem B — Fill color was the mean of ring pixels (wrong on halftone/patterned backgrounds)

#### Symptom
On screentone or patterned backgrounds (e.g. alternating black/white dots), the mean of all ring pixels produced a mid-grey fill rather than the dominant background tone.

#### Fix
Changed from arithmetic mean to **mode**: ring pixels are quantized to the nearest 8-step bucket (`r >> 3`, `g >> 3`, `b >> 3`), then the most frequent bucket wins. The final fill color is the mean of the raw pixels in that winning bucket, giving a precise color from the dominant tone:
```typescript
const key = ((r >> 3) << 10) | ((g >> 3) << 5) | (bv >> 3)
// … accumulate counts and sums per key …
// pick bucket with highest count, average its raw pixels
```

---

## Feature — Text color selector per bubble

**Date:** 2026-03-28
**Affected files:** `src/types/index.ts`, `src/ui/workspace.ts`, `src/ui/workspace.css`, `src/pipeline/typeset.ts`

Added a per-bubble text color option (`black` or `white`) to the editor panel. Default is black (`#1a1a1a` fill, white outline). Selecting white inverts to `#ffffff` fill, `#1a1a1a` outline — useful for dark-background bubbles or background text on black panels.

The field `text_color?: 'black' | 'white'` was added to `MangaBubble`. All four canvas/SVG rendering paths in `typeset.ts` (horizontal text, vertical regular chars, vertical rotated dashes, SVG) read `bubble.text_color` to derive fill and stroke colors.

---

## Issue #11 — Background text fill was rectangular even for bubble-shaped boxes

**Status:** Fixed
**Date:** 2026-04-12
**Affected file:** `src/workers/inpaint.worker.ts`

### Symptom
When a box had both `is_background === true` (Background text checked) and `shape === 'bubble'`, the inpaint fill was still a rectangle, not an ellipse. The rectangular fill extended into the corners outside the oval bubble border, leaving visible filled corners.

### Root cause
The `is_background === true` fill path had no shape awareness at all. It only handled two cases: rotation (rotated rect) and no-rotation (plain axis-aligned rect). The ellipse fill logic existed only in the white route, not in the background/solid route. So bubble-shape was completely ignored when `is_background` was set.

### Fix
Added shape-aware fill to the `is_background === true` path for all four combinations:

- **Rotation + bubble**: iterate rotated AABB, apply ellipse test in local frame (`(lx/rw)² + (ly/rh)² > 1`)
- **Rotation + rect**: existing rotated rect logic
- **No rotation + bubble**: iterate tight rect, apply ellipse test
- **No rotation + rect**: existing axis-aligned rect loop

```typescript
if (b.shape === 'bubble') {
  if ((lx / rw) * (lx / rw) + (ly / rh) * (ly / rh) > 1) continue
} else {
  if (Math.abs(lx) > rw || Math.abs(ly) > rh) continue
}
```

### General lesson
Shape-aware fill logic must be applied consistently across **all** fill routes (white, solid, background). Adding it to one route and not others means the fix only works for specific configurations.

---

## Feature — English / Chinese language switcher

**Date:** 2026-04-27
**Affected files:** `src/i18n/index.ts` (new), `src/ui/workspace.ts`, `src/ui/dict-panel.ts`, `src/ui/settings.ts`, `src/ui/workspace.css`

Added a full EN / 中文 toggle to the topbar. A single `STRINGS` map in `src/i18n/index.ts` holds every UI string keyed by name with both locale variants. `t(key)` returns the current locale's string; `setLocale()` saves to localStorage and fires registered listeners. Static DOM elements use `data-i18n` attributes updated by `applyLocale()`; dynamically-rendered elements (editor panel, bubble list badges, dict panel) are re-rendered in the `onLocaleChange` handler. The toggle is a segmented `[EN][中文]` pill styled in red so it's immediately visible.

---

## Feature — UI polish (2026-04-27)

**Affected files:** `src/ui/workspace.ts`, `src/ui/workspace.css`

- **Text drag always-on**: removed the explicit "Move Text" button; dragging typeset text on the image is now always active. A hint line in the editor panel explains the behaviour.
- **Delete button always visible**: the × button on each bubble list item is now permanently red (`#fc8181`) instead of appearing only on hover.
- **Add Box/Round/Freehand sidebar section**: moved shape selection out of the toolbar into the left sidebar as a gradient button with an attached mode pill showing the current shape and a dropdown to switch between Box, Round, and Freehand.
- **Typeset button colour**: changed from teal to solid purple to match the rest of the action button family.
