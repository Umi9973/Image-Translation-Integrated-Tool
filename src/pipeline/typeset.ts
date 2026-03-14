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

// ── Config ────────────────────────────────────────────────────────────────────

const FONT_FAMILY = "'ZCOOL KuaiLe', 'Microsoft YaHei', 'PingFang SC', sans-serif"
const PADDING     = 6    // px inside bubble rect (SVG user units = natural image px)
const COL_GAP     = 4    // px gap between vertical columns
const MAX_FONT    = 72
const MIN_FONT    = 8

// ── Vertical layout helpers ───────────────────────────────────────────────────

/**
 * Find the largest font size where vertical columns fit inside (maxW × maxH).
 * Each CJK character is approximately fontSize × fontSize (square em).
 * Columns flow top-to-bottom; column order is right-to-left (manga reading order).
 * `segments` is the text split on '\' — each segment starts a new column.
 */
function fitVertical(
  segments: string[],
  maxW: number,
  maxH: number,
): { fontSize: number; charsPerCol: number } {
  const innerW = maxW - PADDING * 2
  const innerH = maxH - PADDING * 2
  const totalChars = segments.reduce((s, seg) => s + seg.length, 0)

  for (let fs = MAX_FONT; fs >= MIN_FONT; fs--) {
    const charsPerCol = Math.max(1, Math.floor(innerH / fs))
    // Each segment gets at least one column; remaining chars spill into extra columns
    let numCols = 0
    for (const seg of segments) {
      numCols += Math.max(1, Math.ceil(seg.length / charsPerCol))
    }
    const totalW = numCols * fs + Math.max(0, numCols - 1) * COL_GAP
    if (totalW <= innerW) return { fontSize: fs, charsPerCol }
  }

  // Fallback: minimum font, may overflow horizontally
  const charsPerCol = Math.max(1, Math.floor(innerH / MIN_FONT))
  return { fontSize: MIN_FONT, charsPerCol }
}

/** Split text on '\' delimiter, stripping empty segments. */
function splitSegments(text: string): string[] {
  return text.split('\\').map(s => s.trim()).filter(s => s.length > 0)
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
export function renderTypeset(bubbles: MangaBubble[], svg: SVGSVGElement): void {
  // Clear previous text
  while (svg.firstChild) svg.removeChild(svg.firstChild)

  // Parse natural dimensions from viewBox
  const vb = svg.getAttribute('viewBox')?.split(' ').map(Number)
  if (!vb || vb.length < 4) return
  const [, , W, H] = vb

  const ns = 'http://www.w3.org/2000/svg'

  for (const bubble of bubbles) {
    const raw = bubble.translated_zh.trim()
    if (!raw) continue

    // Split on '\' — each segment starts a new column group
    const segments = splitSegments(raw)
    if (segments.length === 0) continue

    // Use expanded bubble interior rect if available (set after inpainting),
    // otherwise fall back to the tight text rect from detection.
    const layoutRect = bubble.bubble_rect ?? bubble.rect

    // Convert percentage rect to natural-image pixel coords
    const bx = (layoutRect.x / 100) * W
    const by = (layoutRect.y / 100) * H
    const bw = (layoutRect.w / 100) * W
    const bh = (layoutRect.h / 100) * H

    const { fontSize, charsPerCol } = fitVertical(segments, bw, bh)
    const colStride = fontSize + COL_GAP

    // Count total columns (each segment gets at least one column)
    let numCols = 0
    for (const seg of segments) {
      numCols += Math.max(1, Math.ceil(seg.length / charsPerCol))
    }

    // Total rendered width; center the column block horizontally inside bubble
    const totalW    = numCols * fontSize + Math.max(0, numCols - 1) * COL_GAP
    const blockLeft = bx + (bw - totalW) / 2

    // Rightmost column x (center of column, columns go right-to-left)
    const rightColCenterX = blockLeft + totalW - fontSize / 2
    const topY = by + PADDING

    const g = document.createElementNS(ns, 'g')
    g.setAttribute('font-family',  FONT_FAMILY)
    g.setAttribute('font-size',    String(fontSize))
    // White outline behind black fill — standard manga typesetting look
    g.setAttribute('fill',            '#1a1a1a')
    g.setAttribute('stroke',          'white')
    g.setAttribute('stroke-width',    String(Math.max(1, fontSize * 0.14)))
    g.setAttribute('stroke-linejoin', 'round')
    g.setAttribute('paint-order',     'stroke')

    let colIdx = 0
    for (const seg of segments) {
      const segCols = Math.max(1, Math.ceil(seg.length / charsPerCol))
      for (let c = 0; c < segCols; c++) {
        const colText = seg.slice(c * charsPerCol, (c + 1) * charsPerCol)
        const x = rightColCenterX - colIdx * colStride

        const t = document.createElementNS(ns, 'text')
        t.setAttribute('x',            String(x))
        t.setAttribute('y',            String(topY))
        t.setAttribute('writing-mode', 'vertical-rl')
        t.setAttribute('text-anchor',  'start')   // 'start' = top in vertical mode
        t.textContent = colText
        g.appendChild(t)
        colIdx++
      }
    }

    svg.appendChild(g)
  }
}
