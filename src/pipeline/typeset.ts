/**
 * typeset.ts — Typesetting pipeline stage
 *
 * Renders translated Chinese text onto a dedicated SVG layer.
 * No worker needed — this is pure DOM/SVG manipulation on the main thread.
 *
 * The SVG must have a viewBox matching the natural image dimensions so that
 * percentage-based rect coordinates map to correct pixel positions without
 * distorting the text glyphs.
 *
 * Signature must not change — workspace.ts depends on it.
 */

import type { MangaBubble } from '../types'
import { zhHansModel, Parser } from 'budoux'

// ── Custom BudouX parser (session-scoped, model-clone approach) ───────────────
//
// We keep a mutable deep-clone of the SC model.  When the user adds a no-split
// phrase we inject strong negative BW2 (bigram) scores for every adjacent char
// pair in the phrase — this prevents BudouX from inserting a break there.
//
// BASE-SCORE COMPENSATION
// BudouX computes baseScore = −0.5 × Σ(all model values) once in the Parser
// constructor and adds it to every position's score.  Injecting −6000 into BW2
// lowers the model sum by 6000, raising baseScore by +3000 globally — which
// partially cancels the protection we just added, and fully cancels a second
// phrase's protection after one more addition.
//
// Fix: BudouX only queries 13 specific group names (UW1–6, BW1–3, TW1–4).
// Any other group is included in the sum (and thus baseScore) but never looked
// up during scoring.  We maintain a dummy group 'XX' whose single key tracks
// the cumulative compensation needed to keep the model sum — and therefore
// baseScore — exactly equal to the original unmodified model's sum.
// Each injection that changes the sum by ΔS is offset by −ΔS in XX['∅'],
// so net sum change is always zero regardless of how many phrases are added.

const PHRASE_SCORE = -6000
const COMP_GROUP   = 'XX'   // never queried by BudouX scoring
const COMP_KEY     = '\x00' // single null-byte key, never in real text

type BudouXModel = Record<string, Record<string, number>>

function cloneModel(): BudouXModel {
  return JSON.parse(JSON.stringify(zhHansModel)) as BudouXModel
}

let customModel: BudouXModel = cloneModel()
let zhParser: Parser = new Parser(customModel)

/** Add a no-split phrase to the session parser.
 *  Injects negative BW2 bigram scores and compensates the model sum so that
 *  baseScore stays constant — previous phrases are never affected. */
export function addPhraseToParser(phrase: string): void {
  if (!phrase || phrase.length < 2) return
  if (!customModel['BW2'])       customModel['BW2'] = {}
  if (!customModel[COMP_GROUP])  customModel[COMP_GROUP] = { [COMP_KEY]: 0 }

  let sumChange = 0
  for (let i = 0; i < phrase.length - 1; i++) {
    const bigram = phrase[i] + phrase[i + 1]
    const oldVal = customModel['BW2'][bigram] ?? 0
    const newVal = Math.min(oldVal, PHRASE_SCORE)
    if (newVal !== oldVal) {
      customModel['BW2'][bigram] = newVal
      sumChange += newVal - oldVal   // always negative
    }
  }

  // Offset the sum change so baseScore is unchanged
  if (sumChange !== 0) {
    customModel[COMP_GROUP][COMP_KEY] -= sumChange  // adds a positive value
  }

  zhParser = new Parser(customModel)
}

/** Reset the session parser back to the unmodified default SC model. */
export function resetParser(): void {
  customModel = cloneModel()
  zhParser = new Parser(customModel)
}

// ── Vertical punctuation normalisation ───────────────────────────────────────

/**
 * Replace ASCII punctuation with full-width / CJK equivalents so they render
 * upright in vertical writing mode. Applied before BudouX parsing so layout
 * metrics are consistent.
 *
 * Order matters: multi-char sequences must be replaced before single chars
 * to avoid double-substitution.
 */
const VERTICAL_MAP: [RegExp, string][] = [
  // Expand dot/ellipsis runs into repeated middle dots (U+30FB).
  // Must handle Unicode ellipsis (…, U+2026) FIRST — the LLM outputs these
  // directly. Each … is one glyph that renders as horizontal "..." in canvas
  // (canvas has no writing-mode). Converting to individual ・ chars fixes both
  // the canvas download path and the SVG preview consistently.
  [/…{4}/g,   '・・・・・・・・・・・・'],  // 4× U+2026 = 12 dots
  [/…{3}/g,   '・・・・・・・・・'],        // 3× = 9 dots
  [/…{2}/g,   '・・・・・・'],             // 2× = 6 dots  (most common: ……)
  [/…/g,      '・・・'],                  // 1× = 3 dots
  [/‥/g,      '・・'],                   // U+205F two-dot leader
  // ASCII dot runs (less common from LLM, but handle anyway)
  [/\.{6}/g,  '・・・・・・'],
  [/\.{5}/g,  '・・・・・'],
  [/\.{4}/g,  '・・・・'],
  [/\.{3}/g,  '・・・'],
  [/\.{2}/g,  '・・'],
  [/\./g,     '．'],    // single period → full-width period
  [/,/g,      '，'],    // comma
  [/!/g,      '！'],    // exclamation
  [/\?/g,     '？'],    // question mark
  [/-{2,}/g,  '—' ],   // -- / --- → em dash
  [/-/g,      '－'],    // hyphen  → full-width hyphen
  [/:/g,      '：'],    // colon
  [/;/g,      '；'],    // semicolon
  [/\(/g,     '（'],    // parens
  [/\)/g,     '）'],
  [/~/g,      '～'],
]

function normalizeVertical(text: string): string {
  let s = text
  for (const [re, rep] of VERTICAL_MAP) s = s.replace(re, rep)
  return s
}

// ── Cover background shape helper ────────────────────────────────────────────

function computeRxRy(w: number, h: number, shape: 'rect' | 'bubble' | undefined): { rx: number; ry: number } {
  if (shape === 'bubble') {
    const r = Math.min(w, h) * 0.40
    return { rx: r, ry: r }
  }
  return { rx: 4, ry: 4 }
}

// ── Config ────────────────────────────────────────────────────────────────────

const FONT_FAMILY        = "'ZCOOL KuaiLe', 'Microsoft YaHei', 'PingFang SC', sans-serif"
const PADDING            = 6    // px inside bubble rect (SVG user units = natural image px)
const COL_GAP            = 4    // px gap between vertical columns
const MAX_FONT           = 72
const DOT_RADIUS_FACTOR  = 0.12  // dot radius = fontSize × this
const DOT_STRIDE_FACTOR  = 0.50  // vertical step between dot centres = fontSize × this
const MIN_FONT           = 8

// Title/honorific words that must not start a new column.
// When BudouX splits "名前先生" → ["名前", "先生"], merge them back so the
// name+title stays together. This is a general rule, not case-specific.
const TITLE_WORDS = new Set([
  '先生', '老师', '同学', '老板', '大人', '殿下', '陛下', '阁下',
  '小姐', '女士', '警察', '医生', '博士', '教授',
])

function mergetitles(chunks: string[]): string[] {
  const out: string[] = []
  for (const chunk of chunks) {
    if (out.length > 0 && TITLE_WORDS.has(chunk)) {
      out[out.length - 1] += chunk
    } else {
      out.push(chunk)
    }
  }
  return out
}

// Sentence-final particles / aspect markers that must not open a new column
// on their own. BudouX sometimes splits them off; merge back into the preceding chunk.
// Includes modal particles (吗/吧/…), exclamations, and common aspect markers
// (了/过/着/来/去) that BudouX frequently detaches from short verbs.
const PARTICLES = new Set([
  '吗', '呢', '吧', '啊', '嘛', '哦', '哈', '呀', '哟', '咧', '啦', '喔',
  '唉', '噢', '嗯', '哎', '哇', '呵', '嗨', '哼',
  '了', '过', '着', '来', '去', '得', '地', '的',
])

function mergeparticles(chunks: string[]): string[] {
  const out: string[] = []
  for (const chunk of chunks) {
    if (out.length > 0 && PARTICLES.has(chunk)) {
      out[out.length - 1] += chunk
    } else {
      out.push(chunk)
    }
  }
  return out
}

function isDotRun(s: string): boolean {
  return s.length > 0 && [...s].every(c => c === '・')
}

/**
 * Split chunks that have exactly one dot/non-dot transition:
 *   "word・・・" → ["word", "・・・"]
 *   "・・・word" → ["・・・", "word"]
 * Chunks with 0 or 2+ transitions (e.g. "・・・word・・・") are left whole
 * to preserve deliberate dot-word-dot stylistic patterns.
 * Only called when the initial layout font is below DOT_SPLIT_THRESHOLD.
 */
function splitdots(chunks: string[]): string[] {
  const out: string[] = []
  for (const chunk of chunks) {
    if (!chunk.includes('・')) { out.push(chunk); continue }
    // Count transitions between dot and non-dot characters
    let transitions = 0
    let prevIsDot = chunk[0] === '・'
    for (let i = 1; i < chunk.length; i++) {
      const isDot = chunk[i] === '・'
      if (isDot !== prevIsDot) { transitions++; prevIsDot = isDot }
    }
    if (transitions !== 1) { out.push(chunk); continue }
    // Split at the single transition
    let cur = ''
    prevIsDot = chunk[0] === '・'
    for (const ch of chunk) {
      const isDot = ch === '・'
      if (isDot !== prevIsDot) { out.push(cur); cur = ch; prevIsDot = isDot }
      else cur += ch
    }
    if (cur) out.push(cur)
  }
  return out
}

/**
 * Merge any BudouX-split dot chunks back into one contiguous run.
 * Dot runs are kept whole (never internally split) but are NOT attached to
 * adjacent word chunks — packChunks may place them in their own column.
 */
function mergedots(chunks: string[]): string[] {
  const out: string[] = []
  for (const chunk of chunks) {
    if (out.length > 0 && isDotRun(chunk) && isDotRun(out[out.length - 1])) {
      out[out.length - 1] += chunk
    } else {
      out.push(chunk)
    }
  }
  return out.length > 0 ? out : ['']
}

// ── Vertical layout helpers ───────────────────────────────────────────────────

/**
 * Pack BudouX chunks greedily into columns of `charsPerCol` characters.
 * Chunks are kept whole when possible; only split if a single chunk exceeds
 * the full column height (no choice).
 * Returns an array of column strings (at least one).
 */
function packChunks(
  chunks: string[],
  charsPerCol: number,
  dotThreshold: number,
  forceDotColumn = false,
): { cols: string[]; truncated: boolean } {
  const cols: string[] = []
  let col = ''
  let truncated = false

  const pushDot = (chunk: string) => {
    if (chunk.length > dotThreshold) { cols.push(chunk.slice(0, dotThreshold)); truncated = true }
    else cols.push(chunk)
  }

  for (const chunk of chunks) {
    // forceDotColumn: pure dot runs always get their own column so the font
    // algorithm can size word columns independently of the dot run length.
    if (forceDotColumn && isDotRun(chunk)) {
      if (col) { cols.push(col); col = '' }
      pushDot(chunk)
      continue
    }
    if (col.length + chunk.length <= charsPerCol) {
      col += chunk
    } else {
      if (col) { cols.push(col); col = '' }
      if (chunk.includes('・')) {
        pushDot(chunk)
      } else {
        // Regular chunk longer than a full column — must hard-split it.
        // Never split immediately before a sentence-final particle so that
        // e.g. "是吗" stays together even when "是" is the last char of a slice.
        let rem = chunk
        while (rem.length > charsPerCol) {
          let splitAt = charsPerCol
          if (splitAt < rem.length && PARTICLES.has(rem[splitAt])) splitAt++
          cols.push(rem.slice(0, splitAt))
          rem = rem.slice(splitAt)
        }
        col = rem
      }
    }
  }
  if (col) cols.push(col)
  return { cols: cols.length > 0 ? cols : [''], truncated }
}

/**
 * Find the largest font size where all columns fit inside (maxW × maxH).
 * Each segment (split on '\') is parsed into BudouX chunks independently,
 * then packed greedily. Returns the font size and the packed columns per segment.
 */
// Max dots that fit exactly within the bubble height (no overflow).
// Rendering is also clipped to the bubble rect, so this is just the packing limit.
const DOT_OVERFLOW_FACTOR = 1.0

function fitVertical(
  segChunks: string[][],
  maxW: number,
  maxH: number,
  forceDotColumn = false,
  forcedFontSize?: number,
): { fontSize: number; segColumns: string[][]; truncated: boolean } {
  const innerW = maxW - PADDING * 2
  const innerH = maxH - PADDING * 2

  // Forced font: skip auto-fit, use the exact size the user chose.
  if (forcedFontSize !== undefined) {
    const fs = Math.max(MIN_FONT, forcedFontSize)
    const charsPerCol  = Math.max(1, Math.floor(innerH / fs))
    const dotThreshold = Math.max(1, Math.ceil(innerH * DOT_OVERFLOW_FACTOR / (fs * DOT_STRIDE_FACTOR)))
    const packed = segChunks.map(chunks => packChunks(chunks, charsPerCol, dotThreshold, forceDotColumn))
    return {
      fontSize:   fs,
      segColumns: packed.map(p => p.cols),
      truncated:  packed.some(p => p.truncated),
    }
  }

  // Longest chunk that must not be hard-split across columns.
  // Protects two categories:
  //   1. Short chunks ≤ MAX_WORD_LEN (regular words like 还, 在, 为什么)
  //   2. Any chunk ending with a title word (name+title merged by mergetitles,
  //      e.g. "...奈良先生" — must stay whole regardless of total length)
  const MAX_WORD_LEN = 4
  const maxChunkLen = segChunks.flat().reduce((m, c) => {
    if (c.includes('・')) return m  // dot columns never force font shrink
    const isShortWord   = c.length <= MAX_WORD_LEN
    const endsWithTitle = [...TITLE_WORDS].some(t => c.endsWith(t))
    return (isShortWord || endsWithTitle) ? Math.max(m, c.length) : m
  }, 1)

  // Standard pass: largest font whose columns fit the bubble width.
  // dotThreshold is computed per-fs since DOT_STRIDE now scales with fontSize.
  let best: { fontSize: number; segColumns: string[][]; truncated: boolean } | null = null
  for (let fs = MAX_FONT; fs >= MIN_FONT; fs--) {
    const charsPerCol  = Math.max(1, Math.floor(innerH / fs))
    if (charsPerCol < maxChunkLen) continue
    const dotThreshold = Math.max(1, Math.ceil(innerH * DOT_OVERFLOW_FACTOR / (fs * DOT_STRIDE_FACTOR)))
    const packed       = segChunks.map(chunks => packChunks(chunks, charsPerCol, dotThreshold, forceDotColumn))
    const segColumns   = packed.map(p => p.cols)
    const truncated    = packed.some(p => p.truncated)
    const numCols      = segColumns.reduce((s, cols) => s + cols.length, 0)
    const totalW       = numCols * fs + Math.max(0, numCols - 1) * COL_GAP
    if (totalW <= innerW) { best = { fontSize: fs, segColumns, truncated }; break }
  }

  // Single-column preference for short text.
  // Guards: single segment (no '\' break), total chars ≤ 5, bubble taller than wide,
  // result font ≥ 12px, and maxChunkLen respected.
  const totalChars = segChunks.flat().reduce((s, c) => s + (c.includes('・') ? 0 : c.length), 0)
  if (
    best !== null &&
    best.segColumns.reduce((s, cols) => s + cols.length, 0) > 1 &&
    segChunks.length === 1 &&
    totalChars <= 5 &&
    innerH > innerW
  ) {
    for (let fs = best.fontSize - 1; fs >= Math.max(MIN_FONT, 12); fs--) {
      const charsPerCol  = Math.max(1, Math.floor(innerH / fs))
      if (charsPerCol < maxChunkLen) continue
      const dotThreshold = Math.max(1, Math.ceil(innerH * DOT_OVERFLOW_FACTOR / (fs * DOT_STRIDE_FACTOR)))
      const packed       = segChunks.map(chunks => packChunks(chunks, charsPerCol, dotThreshold, forceDotColumn))
      const segColumns   = packed.map(p => p.cols)
      const truncated    = packed.some(p => p.truncated)
      const numCols      = segColumns.reduce((s, cols) => s + cols.length, 0)
      const totalW       = numCols * fs + Math.max(0, numCols - 1) * COL_GAP
      if (totalW <= innerW && numCols === 1) return { fontSize: fs, segColumns, truncated }
    }
  }

  if (best) return best

  // Fallback: minimum font, may overflow — still respect maxChunkLen to avoid hard-splits
  const charsPerCol  = Math.max(maxChunkLen, Math.max(1, Math.floor(innerH / MIN_FONT)))
  const dotThreshold = Math.max(1, Math.ceil(innerH * DOT_OVERFLOW_FACTOR / (MIN_FONT * DOT_STRIDE_FACTOR)))
  const packed       = segChunks.map(chunks => packChunks(chunks, charsPerCol, dotThreshold, forceDotColumn))
  return {
    fontSize:   MIN_FONT,
    segColumns: packed.map(p => p.cols),
    truncated:  packed.some(p => p.truncated),
  }
}

/** Split text on '\' delimiter, stripping empty segments. */
function splitSegments(text: string): string[] {
  return text.split('\\').map(s => s.trim()).filter(s => s.length > 0)
}

/** Rendered pixel height of one column string (dots use DOT_STRIDE_FACTOR × fontSize, text uses fontSize). */
function columnHeight(col: string, fontSize: number): number {
  const dotStride = fontSize * DOT_STRIDE_FACTOR
  let h = 0
  for (const ch of col) h += ch === '・' ? dotStride : fontSize
  return h
}

/** Max rendered height across all columns in all segments. */
function maxColumnHeight(segColumns: string[][], fontSize: number): number {
  let max = 0
  for (const cols of segColumns)
    for (const col of cols)
      max = Math.max(max, columnHeight(col, fontSize))
  return max
}

// ── Canvas export (font-correct download) ────────────────────────────────────

/**
 * Render typeset text directly onto a Canvas 2D context.
 * Uses the same layout logic as renderTypeset but draws via Canvas API so that
 * fonts already loaded in the page (Google Fonts) are available — unlike
 * SVG-as-<img> which blocks external resources.
 *
 * @param bubbles  Bubbles to render (skips any with empty translated_zh)
 * @param ctx      Canvas 2D rendering context
 * @param W        Canvas / natural image width in pixels
 * @param H        Canvas / natural image height in pixels
 */
export function renderTypesetToCanvas(
  bubbles: MangaBubble[],
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
): void {
  for (const bubble of bubbles) {
    const raw = bubble.translated_zh.trim()
    if (!raw) continue

    const layoutRect = bubble.bubble_rect ?? bubble.rect
    const bx = (layoutRect.x / 100) * W
    const by = (layoutRect.y / 100) * H
    const bw = (layoutRect.w / 100) * W
    const bh = (layoutRect.h / 100) * H

    // ── Horizontal text path ─────────────────────────────────────────────────
    if (bubble.text_direction === 'horizontal') {
      const lines = raw.split('\\').map(l => l.trim()).filter(Boolean)
      const innerW = bw - PADDING * 2
      const innerH = bh - PADDING * 2
      const override = bubble.font_size_override
      let fontSize = MIN_FONT
      if (override !== undefined) {
        fontSize = Math.max(MIN_FONT, override)
      } else {
        for (let fs = MAX_FONT; fs >= MIN_FONT; fs--) {
          ctx.font = `${fs}px ${FONT_FAMILY}`
          const maxLineW = Math.max(...lines.map(l => ctx.measureText(l).width))
          if (maxLineW <= innerW && lines.length * fs * 1.2 <= innerH) { fontSize = fs; break }
        }
      }
      if (bubble.cover || bubble.source === 'manual') {
        const { rx, ry } = computeRxRy(bw, bh, bubble.shape)
        const r = Math.min(rx, ry)
        const strokeW = Math.max(2, Math.min(bw, bh) * 0.025)
        ctx.save()
        ctx.beginPath()
        if (typeof (ctx as CanvasRenderingContext2D & { roundRect?: (...a: unknown[]) => void }).roundRect === 'function') {
          (ctx as CanvasRenderingContext2D & { roundRect: (x: number, y: number, w: number, h: number, r: number) => void }).roundRect(bx, by, bw, bh, r)
        } else {
          ctx.rect(bx, by, bw, bh)
        }
        ctx.fillStyle = '#ffffff'
        ctx.fill()
        if (bubble.coverOutline) { ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = strokeW; ctx.stroke() }
        ctx.restore()
      }
      ctx.save()
      ctx.beginPath(); ctx.rect(bx, by, bw, bh); ctx.clip()
      ctx.font = `${fontSize}px ${FONT_FAMILY}`
      ctx.textAlign    = 'center'
      ctx.textBaseline = 'middle'
      const strokeWidth = Math.max(1, fontSize * 0.14)
      const lineH  = fontSize * 1.2
      const totalH = lines.length * lineH
      const startY = by + (bh - totalH) / 2 + lineH / 2
      const cx = bx + bw / 2
      lines.forEach((line, i) => {
        ctx.strokeStyle = 'white'; ctx.lineWidth = strokeWidth * 2; ctx.lineJoin = 'round'
        ctx.strokeText(line, cx, startY + i * lineH)
        ctx.fillStyle = '#1a1a1a'
        ctx.fillText(line, cx, startY + i * lineH)
      })
      ctx.restore()
      continue
    }

    const segments = splitSegments(raw)
    if (segments.length === 0) continue

    const segChunks = segments.map(seg => {
      const normalized = normalizeVertical(seg)
      return mergedots(mergeparticles(mergetitles(zhParser.parse(normalized))))
    })

    const override = bubble.font_size_override
    let { fontSize, segColumns, truncated: trunc0 } = fitVertical(segChunks, bw, bh, false, override)
    const alt = fitVertical(segChunks.map(splitdots), bw, bh, true, override)
    if (alt.fontSize > fontSize || (alt.fontSize === fontSize && !alt.truncated && trunc0)) {
      ;({ fontSize, segColumns } = alt)
    }
    const colStride = fontSize + COL_GAP

    const numCols   = segColumns.reduce((s, cols) => s + cols.length, 0)
    const totalW    = numCols * fontSize + Math.max(0, numCols - 1) * COL_GAP
    const blockLeft = bx + (bw - totalW) / 2

    const rightColCenterX = blockLeft + totalW - fontSize / 2
    const textH = maxColumnHeight(segColumns, fontSize)
    const topY  = Math.max(by + PADDING, by + (bh - textH) / 2)

    if (bubble.cover || bubble.source === 'manual') {
      const { rx, ry } = computeRxRy(bw, bh, bubble.shape)
      const r = Math.min(rx, ry)
      const strokeW = Math.max(2, Math.min(bw, bh) * 0.025)
      ctx.save()
      ctx.beginPath()
      if (typeof (ctx as CanvasRenderingContext2D & { roundRect?: (...a: unknown[]) => void }).roundRect === 'function') {
        (ctx as CanvasRenderingContext2D & { roundRect: (x: number, y: number, w: number, h: number, r: number) => void }).roundRect(bx, by, bw, bh, r)
      } else {
        ctx.rect(bx, by, bw, bh)
      }
      ctx.fillStyle = '#ffffff'
      ctx.fill()
      if (bubble.coverOutline) {
        ctx.strokeStyle = '#1a1a1a'
        ctx.lineWidth = strokeW
        ctx.stroke()
      }
      ctx.restore()
    }

    ctx.save()
    // Clip canvas to bubble rect so dots/text never bleed outside
    ctx.beginPath()
    ctx.rect(bx, by, bw, bh)
    ctx.clip()
    ctx.font         = `${fontSize}px ${FONT_FAMILY}`
    ctx.textBaseline = 'top'
    ctx.textAlign    = 'center'

    const strokeWidth = Math.max(1, fontSize * 0.14)
    const dotRadius   = fontSize * DOT_RADIUS_FACTOR
    const dotStride   = fontSize * DOT_STRIDE_FACTOR

    // Glyphs that are horizontal by nature and need 90° rotation in vertical
    // layout. SVG writing-mode handles this automatically; canvas does not.
    const ROTATE_CHARS = new Set(['－', '—', '–', '―', '─'])

    let colIdx = 0
    for (const cols of segColumns) {
      for (const colText of cols) {
        const x = rightColCenterX - colIdx * colStride

        let y = topY
        for (const ch of colText) {
          const cx = x

          if (ch === '・') {
            const dotCy  = y + dotStride * 0.5
            const outerR = Math.min(dotStride * 0.45, dotRadius + strokeWidth * 0.5)
            ctx.save()
            ctx.beginPath()
            ctx.arc(cx, dotCy, outerR, 0, Math.PI * 2)
            ctx.fillStyle = 'white'
            ctx.fill()
            ctx.beginPath()
            ctx.arc(cx, dotCy, dotRadius, 0, Math.PI * 2)
            ctx.fillStyle = '#1a1a1a'
            ctx.fill()
            ctx.restore()
            y += dotStride
          } else if (ROTATE_CHARS.has(ch)) {
            // Rotate 90° around the centre of the character cell so horizontal
            // dash glyphs become vertical strokes, matching SVG behaviour.
            ctx.save()
            ctx.translate(cx, y + fontSize * 0.5)
            ctx.rotate(Math.PI / 2)
            ctx.strokeStyle = 'white'
            ctx.lineWidth   = strokeWidth * 2
            ctx.lineJoin    = 'round'
            ctx.strokeText(ch, 0, -fontSize * 0.5)
            ctx.fillStyle = '#1a1a1a'
            ctx.fillText(ch, 0, -fontSize * 0.5)
            ctx.restore()
            y += fontSize
          } else {
            ctx.strokeStyle   = 'white'
            ctx.lineWidth     = strokeWidth * 2
            ctx.lineJoin      = 'round'
            ctx.strokeText(ch, cx, y)
            ctx.fillStyle = '#1a1a1a'
            ctx.fillText(ch, cx, y)
            y += fontSize
          }
        }

        colIdx++
      }
    }

    ctx.restore()
  }
}

// ── Main export ───────────────────────────────────────────────────────────────

/**
 * Render translated text for all bubbles onto the typeset SVG layer.
 * Clears any previous typesetting before re-rendering.
 *
 * @param bubbles  Bubbles to render (skips any with empty translated_zh)
 * @param svg      The `.ws-typeset-layer` SVG element; must have viewBox set
 *                 to "0 0 <naturalWidth> <naturalHeight>"
 */
/**
 * Returns the IDs of bubbles where dot runs were silently clipped because
 * they exceeded DOT_OVERFLOW_FACTOR × the available column height.
 */
export function renderTypeset(
  bubbles: MangaBubble[],
  svg: SVGSVGElement,
): { clippedIds: string[]; fontSizes: Record<string, number> } {
  // Clear previous text
  while (svg.firstChild) svg.removeChild(svg.firstChild)

  const clippedIds: string[] = []
  const fontSizes: Record<string, number> = {}
  const debugLog: object[] = []

  // Parse natural dimensions from viewBox
  const vb = svg.getAttribute('viewBox')?.split(' ').map(Number)
  if (!vb || vb.length < 4) return { clippedIds, fontSizes }
  const [, , W, H] = vb

  const ns = 'http://www.w3.org/2000/svg'

  for (const bubble of bubbles) {
    const raw = bubble.translated_zh.trim()
    if (!raw) continue

    // Use expanded bubble interior rect if available (set after inpainting),
    // otherwise fall back to the tight text rect from detection.
    const layoutRect = bubble.bubble_rect ?? bubble.rect

    // Convert percentage rect to natural-image pixel coords
    const bx = (layoutRect.x / 100) * W
    const by = (layoutRect.y / 100) * H
    const bw = (layoutRect.w / 100) * W
    const bh = (layoutRect.h / 100) * H

    // ── Horizontal text path ─────────────────────────────────────────────────
    if (bubble.text_direction === 'horizontal') {
      const lines = raw.split('\\').map(l => l.trim()).filter(Boolean)
      const innerW = bw - PADDING * 2
      const innerH = bh - PADDING * 2
      const override = bubble.font_size_override
      let fontSize = MIN_FONT
      if (override !== undefined) {
        fontSize = Math.max(MIN_FONT, override)
      } else {
        for (let fs = MAX_FONT; fs >= MIN_FONT; fs--) {
          if ([...lines].every(l => [...l].length * fs * 0.95 <= innerW) && lines.length * fs * 1.2 <= innerH) {
            fontSize = fs; break
          }
        }
      }
      fontSizes[bubble.id] = fontSize

      const bubbleGroup = document.createElementNS(ns, 'g')
      bubbleGroup.setAttribute('data-bubble-id', bubble.id)

      if (bubble.cover || bubble.source === 'manual') {
        const { rx, ry } = computeRxRy(bw, bh, bubble.shape)
        const strokeW = Math.max(2, Math.min(bw, bh) * 0.025)
        const bg = document.createElementNS(ns, 'rect')
        bg.setAttribute('x', String(bx)); bg.setAttribute('y', String(by))
        bg.setAttribute('width', String(bw)); bg.setAttribute('height', String(bh))
        bg.setAttribute('rx', String(rx)); bg.setAttribute('ry', String(ry))
        bg.setAttribute('fill', '#ffffff')
        if (bubble.coverOutline) { bg.setAttribute('stroke', '#1a1a1a'); bg.setAttribute('stroke-width', String(strokeW)) }
        else bg.setAttribute('stroke', 'none')
        bubbleGroup.appendChild(bg)
      }

      const clipId = `bc-${bubble.id}`
      const clipPath = document.createElementNS(ns, 'clipPath')
      clipPath.setAttribute('id', clipId)
      const clipRect = document.createElementNS(ns, 'rect')
      clipRect.setAttribute('x', String(bx)); clipRect.setAttribute('y', String(by))
      clipRect.setAttribute('width', String(bw)); clipRect.setAttribute('height', String(bh))
      clipPath.appendChild(clipRect)
      bubbleGroup.appendChild(clipPath)

      const g = document.createElementNS(ns, 'g')
      g.setAttribute('clip-path', `url(#${clipId})`)
      g.setAttribute('font-family', FONT_FAMILY)
      g.setAttribute('font-size', String(fontSize))
      g.setAttribute('fill', '#1a1a1a')
      g.setAttribute('stroke', 'white')
      g.setAttribute('stroke-width', String(Math.max(1, fontSize * 0.14)))
      g.setAttribute('stroke-linejoin', 'round')
      g.setAttribute('paint-order', 'stroke')
      g.setAttribute('text-anchor', 'middle')
      g.setAttribute('dominant-baseline', 'middle')

      const lineH  = fontSize * 1.2
      const totalH = lines.length * lineH
      const startY = by + (bh - totalH) / 2 + lineH / 2
      const cx = bx + bw / 2

      lines.forEach((line, i) => {
        const t = document.createElementNS(ns, 'text')
        t.setAttribute('x', String(cx))
        t.setAttribute('y', String(startY + i * lineH))
        t.textContent = line
        g.appendChild(t)
      })

      bubbleGroup.appendChild(g)
      svg.appendChild(bubbleGroup)
      continue
    }

    // Split on '\' — each segment starts a new column group
    const segments = splitSegments(raw)
    if (segments.length === 0) continue

    // Parse each segment into BudouX chunks, then merge title words so
    // name+title (e.g. 奈良先生) stays in one column.
    const segChunks = segments.map(seg => {
      const normalized = normalizeVertical(seg)
      return mergedots(mergeparticles(mergetitles(zhParser.parse(normalized))))
    })

    const override = bubble.font_size_override
    const initialFit = fitVertical(segChunks, bw, bh, false, override)
    let { fontSize, segColumns, truncated } = initialFit
    const altFit = fitVertical(segChunks.map(splitdots), bw, bh, true, override)
    const splitDotsApplied = altFit.fontSize > fontSize || (altFit.fontSize === fontSize && !altFit.truncated && truncated)
    if (splitDotsApplied) ({ fontSize, segColumns, truncated } = altFit)
    fontSizes[bubble.id] = fontSize
    if (truncated) clippedIds.push(bubble.id)
    const colStride = fontSize + COL_GAP

    const numCols = segColumns.reduce((s, cols) => s + cols.length, 0)
    const totalW    = numCols * fontSize + Math.max(0, numCols - 1) * COL_GAP
    const blockLeft = bx + (bw - totalW) / 2

    // Rightmost column x (center of column, columns go right-to-left)
    const rightColCenterX = blockLeft + totalW - fontSize / 2
    const textH = maxColumnHeight(segColumns, fontSize)
    const topY  = Math.max(by + PADDING, by + (bh - textH) / 2)

    debugLog.push({
      id:               bubble.id,
      text:             raw,
      rectSource:       bubble.bubble_rect ? 'bubble_rect' : 'rect',
      bubblePx:         { w: Math.round(bw), h: Math.round(bh) },
      chunksInitial:    segChunks.map(c => c.join(' | ')),
      initialFontSize:  initialFit.fontSize,
      splitDotsApplied,
      chunksAfterSplit: splitDotsApplied ? segChunks.map(c => splitdots(c).join(' | ')) : null,
      fontSize,
      numCols,
      columns:          segColumns.flat(),
      textHeightPx:     Math.round(textH),
      verticalOffset:   Math.round(topY - by),
      centered:         topY > by + PADDING,
      truncated,
    })

    // Wrapper group — lets workspace remove one bubble's typeset by data-bubble-id
    const bubbleGroup = document.createElementNS(ns, 'g')
    bubbleGroup.setAttribute('data-bubble-id', bubble.id)

    if (bubble.cover || bubble.source === 'manual') {
      const { rx, ry } = computeRxRy(bw, bh, bubble.shape)
      const strokeW = Math.max(2, Math.min(bw, bh) * 0.025)
      const bg = document.createElementNS(ns, 'rect')
      bg.setAttribute('x',      String(bx))
      bg.setAttribute('y',      String(by))
      bg.setAttribute('width',  String(bw))
      bg.setAttribute('height', String(bh))
      bg.setAttribute('rx',     String(rx))
      bg.setAttribute('ry',     String(ry))
      bg.setAttribute('fill',   '#ffffff')
      if (bubble.coverOutline) {
        bg.setAttribute('stroke',       '#1a1a1a')
        bg.setAttribute('stroke-width', String(strokeW))
      } else {
        bg.setAttribute('stroke', 'none')
      }
      bubbleGroup.appendChild(bg)
    }

    // Clip group to bubble rect so dots and text never bleed outside
    const clipId = `bc-${bubble.id}`
    const clipPath = document.createElementNS(ns, 'clipPath')
    clipPath.setAttribute('id', clipId)
    const clipRect = document.createElementNS(ns, 'rect')
    clipRect.setAttribute('x',      String(bx))
    clipRect.setAttribute('y',      String(by))
    clipRect.setAttribute('width',  String(bw))
    clipRect.setAttribute('height', String(bh))
    clipPath.appendChild(clipRect)
    bubbleGroup.appendChild(clipPath)

    const g = document.createElementNS(ns, 'g')
    g.setAttribute('clip-path',    `url(#${clipId})`)
    g.setAttribute('font-family',  FONT_FAMILY)
    g.setAttribute('font-size',    String(fontSize))
    // White outline behind black fill — standard manga typesetting look
    g.setAttribute('fill',            '#1a1a1a')
    g.setAttribute('stroke',          'white')
    g.setAttribute('stroke-width',    String(Math.max(1, fontSize * 0.14)))
    g.setAttribute('stroke-linejoin', 'round')
    g.setAttribute('paint-order',     'stroke')

    const dotRadius = fontSize * DOT_RADIUS_FACTOR
    const dotStride = fontSize * DOT_STRIDE_FACTOR

    let colIdx = 0
    for (const cols of segColumns) {
      for (const colText of cols) {
        const x = rightColCenterX - colIdx * colStride

        let y        = topY
        let textSeg  = ''
        let textSegY = topY

        const flushText = () => {
          if (!textSeg) return
          const t = document.createElementNS(ns, 'text')
          t.setAttribute('x',            String(x))
          t.setAttribute('y',            String(textSegY))
          t.setAttribute('writing-mode', 'vertical-rl')
          t.setAttribute('text-anchor',  'start')
          t.textContent = textSeg
          g.appendChild(t)
          textSeg = ''
        }

        for (const ch of colText) {
          if (ch === '・') {
            flushText()
            const circle = document.createElementNS(ns, 'circle')
            circle.setAttribute('cx', String(x))
            circle.setAttribute('cy', String(y + dotStride * 0.5))
            circle.setAttribute('r',  String(dotRadius))
            g.appendChild(circle)
            y       += dotStride
            textSegY = y
          } else {
            textSeg += ch
            y       += fontSize
          }
        }
        flushText()

        colIdx++
      }
    }

    bubbleGroup.appendChild(g)
    svg.appendChild(bubbleGroup)
  }

  fetch('/__debug/typeset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(debugLog),
  }).catch(() => {})
  return { clippedIds, fontSizes }
}
