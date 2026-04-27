import './workspace.css'
import type { MangaBubble, MangaPage, BubbleState } from '../types'
import { detectBubblesWithMask } from '../pipeline/detect'
import type { DetectionMask } from '../pipeline/detect'
import { runOCR } from '../pipeline/ocr'
import { buildPrompt, parseTranslationResponse } from '../pipeline/translate'
import { inpaintPage } from '../pipeline/inpaint'
import { renderTypeset, renderTypesetToCanvas, parseRuby } from '../pipeline/typeset'
import type { RubySpan } from '../pipeline/typeset'
import { openSettings } from './settings'
import { renderDictPanel, getGlossary } from './dict-panel'
import { t, setLocale, getLocale, onLocaleChange, applyLocale } from '../i18n'
import type { I18nKey } from '../i18n'

// ── Constants ────────────────────────────────────────────────────────────────

const ROW_THRESHOLD = 5 // bubble y-values within 5% are considered the same row

const STATE_KEY: Record<BubbleState, I18nKey> = {
  detected:   'stateDetected',
  ocr_done:   'stateOcrDone',
  translated: 'stateTranslated',
  reviewed:   'stateReviewed',
}
function stateLabel(state: BubbleState): string { return t(STATE_KEY[state]) }

// ── Helpers ───────────────────────────────────────────────────────────────────


function truncate(str: string, max: number): string {
  return str.length <= max ? str : str.slice(0, max) + '…'
}

function makeBadge(state: BubbleState, extraClass = ''): HTMLSpanElement {
  const el = document.createElement('span')
  el.className = `ws-badge ws-badge--${state}${extraClass ? ' ' + extraClass : ''}`
  el.textContent = stateLabel(state)
  return el
}

function sortBubbleIds(bubbles: MangaBubble[]): string[] {
  return [...bubbles]
    .sort((a, b) => {
      const rowA = Math.round(a.rect.y / ROW_THRESHOLD)
      const rowB = Math.round(b.rect.y / ROW_THRESHOLD)
      if (rowA !== rowB) return rowA - rowB   // top rows first
      return b.rect.x - a.rect.x              // rightmost first (RTL)
    })
    .map(b => b.id)
}

// ── Surgical DOM updaters (avoid full re-renders on simple changes) ───────────

function updateSvgSelection(svg: SVGSVGElement, id: string, selected: boolean): void {
  svg.querySelector(`[data-id="${id}"]`)?.classList.toggle('is-selected', selected)
}

function updateListSelection(list: HTMLElement, id: string, selected: boolean): void {
  list.querySelector(`[data-id="${id}"]`)?.classList.toggle('is-selected', selected)
}

function updateListPreview(list: HTMLElement, id: string, text: string): void {
  const el = list.querySelector<HTMLElement>(`[data-preview="${id}"]`)
  if (el) el.textContent = truncate(text, 18)
}

function updateListBadge(list: HTMLElement, id: string, state: BubbleState): void {
  const el = list.querySelector<HTMLElement>(`[data-badge="${id}"]`)
  if (!el) return
  el.className = `ws-badge ws-badge--${state}`
  el.textContent = stateLabel(state)
}

function updateEditorBadge(editor: HTMLElement, state: BubbleState): void {
  const el = editor.querySelector<HTMLElement>('.ws-editor-badge')
  if (!el) return
  el.className = `ws-badge ws-badge--${state} ws-editor-badge`
  el.textContent = stateLabel(state)
}

// ── Editor rendering ──────────────────────────────────────────────────────────

interface EditorCallbacks {
  onTextChange:  (field: 'raw_ja' | 'translated_zh', value: string) => void
  onLockToggle:  () => void
  onNavigate:    (direction: -1 | 1) => void
  onCoverChange:    (cover: boolean) => void
  onOutlineChange:  (outline: boolean) => void
  onShapeChange:    (shape: 'rect' | 'bubble') => void
  onRevertInpaint?:  () => void
  onRevertTypeset?:  () => void
  onFontSizeOverrideChange: (size: number | undefined) => void
  onDirectionChange: (dir: 'vertical' | 'horizontal') => void
  onResetPosition: () => void
  onIsBackgroundChange: (val: boolean) => void
  onTextColorChange: (color: 'black' | 'white') => void
  onRotationChange: (deg: number | undefined) => void
}

// ── Ruby panel ────────────────────────────────────────────────────────────────

/** Rebuild translated_zh from clean text + ruby spans (sorted by start). */
function buildRubyText(clean: string, spans: RubySpan[]): string {
  const sorted = [...spans].sort((a, b) => a.start - b.start)
  let result = ''
  let pos = 0
  for (const span of sorted) {
    result += clean.slice(pos, span.start)
    result += `{${clean.slice(span.start, span.end)}|${span.ruby}}`
    pos = span.end
  }
  return result + clean.slice(pos)
}

/**
 * Render the ruby annotation panel below the ZH textarea.
 * Shows each character as a clickable chip; existing ruby spans are highlighted.
 * Selecting chips and typing in the input adds/replaces ruby annotations.
 * Returns the panel element to append into the editor.
 */
function renderRubyPanel(
  initialText: string,
  onChange: (newText: string) => void,
): HTMLElement {
  const section = document.createElement('div')
  section.className = 'ws-ruby-section'

  const label = document.createElement('div')
  label.className = 'ws-editor-label'
  label.textContent = 'Ruby Annotations'
  section.appendChild(label)

  const chipsContainer = document.createElement('div')
  chipsContainer.className = 'ws-ruby-chips'
  section.appendChild(chipsContainer)

  const inputRow = document.createElement('div')
  inputRow.className = 'ws-ruby-input-row'
  inputRow.hidden = true

  const rubyInput = document.createElement('input')
  rubyInput.type = 'text'
  rubyInput.className = 'ws-ruby-input'
  rubyInput.placeholder = 'ruby text…'

  const confirmBtn = document.createElement('button')
  confirmBtn.type = 'button'
  confirmBtn.className = 'ws-ruby-confirm'
  confirmBtn.textContent = '✓'
  confirmBtn.title = 'Apply ruby annotation'

  const cancelBtn = document.createElement('button')
  cancelBtn.type = 'button'
  cancelBtn.className = 'ws-ruby-cancel'
  cancelBtn.textContent = '×'
  cancelBtn.title = 'Cancel'

  inputRow.appendChild(rubyInput)
  inputRow.appendChild(confirmBtn)
  inputRow.appendChild(cancelBtn)
  section.appendChild(inputRow)

  let selected = new Set<number>()  // selected char indices in clean text
  let currentText = initialText

  function rebuild(text: string) {
    currentText = text
    chipsContainer.innerHTML = ''
    selected.clear()
    inputRow.hidden = true
    rubyInput.value = ''

    const { clean, spans } = parseRuby(text)

    // Map charIndex → span that contains it
    const charSpan = new Map<number, RubySpan>()
    for (const span of spans) {
      for (let i = span.start; i < span.end; i++) charSpan.set(i, span)
    }

    let globalIdx = 0
    // Split on \\ to show segment breaks visually
    const segments = clean.split('\\')
    segments.forEach((seg, si) => {
      for (let ci = 0; ci < seg.length; ci++) {
        const ch = seg[ci]
        const absIdx = globalIdx
        const span = charSpan.get(absIdx)

        const chip = document.createElement('div')
        chip.className = 'ws-ruby-chip'
        chip.dataset.idx = String(absIdx)
        if (span) {
          chip.classList.add('has-ruby')
          // Only show ruby label on the first char of the span
          if (span.start === absIdx) {
            chip.dataset.spanStart = 'true'
            chip.dataset.spanEnd   = String(span.end - 1)
            const rubyEl = document.createElement('span')
            rubyEl.className = 'ws-ruby-chip-ruby'
            rubyEl.textContent = span.ruby
            chip.appendChild(rubyEl)
          }
        }
        const mainEl = document.createElement('span')
        mainEl.className = 'ws-ruby-chip-main'
        mainEl.textContent = ch
        chip.appendChild(mainEl)

        chip.addEventListener('click', () => {
          // If clicking a char that belongs to an existing span, select entire span
          const targetSpan = charSpan.get(absIdx)
          if (targetSpan && !selected.has(absIdx)) {
            selected.clear()
            for (let i = targetSpan.start; i < targetSpan.end; i++) selected.add(i)
            rubyInput.value = targetSpan.ruby
          } else if (selected.has(absIdx)) {
            selected.delete(absIdx)
          } else {
            selected.add(absIdx)
            rubyInput.value = ''
          }
          refreshSelection()
        })

        chipsContainer.appendChild(chip)
        globalIdx++
      }

      // Show segment break between segments (not after last)
      if (si < segments.length - 1) {
        const sep = document.createElement('div')
        sep.className = 'ws-ruby-chip-sep'
        sep.textContent = '↵'
        chipsContainer.appendChild(sep)
        globalIdx++ // account for the '\\' char in clean text
      }
    })
  }

  function refreshSelection() {
    chipsContainer.querySelectorAll<HTMLElement>('.ws-ruby-chip').forEach(chip => {
      const idx = Number(chip.dataset.idx)
      chip.classList.toggle('is-selected', selected.has(idx))
    })
    inputRow.hidden = selected.size === 0
    if (selected.size > 0) rubyInput.focus()
  }

  function applyRuby() {
    const ruby = rubyInput.value.trim()
    if (selected.size === 0) return

    const { clean, spans } = parseRuby(currentText)
    const selArr = [...selected].sort((a, b) => a - b)
    const selStart = selArr[0], selEnd = selArr[selArr.length - 1] + 1

    // Remove any spans that overlap the selection
    const filtered = spans.filter(s => s.end <= selStart || s.start >= selEnd)

    // Add new span only if ruby text was provided
    const newSpans = ruby.length > 0
      ? [...filtered, { start: selStart, end: selEnd, ruby }]
      : filtered

    const newText = buildRubyText(clean, newSpans)
    onChange(newText)
    rebuild(newText)
  }

  confirmBtn.addEventListener('click', applyRuby)
  rubyInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); applyRuby() }
    if (e.key === 'Escape') { selected.clear(); refreshSelection() }
  })
  cancelBtn.addEventListener('click', () => { selected.clear(); refreshSelection() })

  rebuild(initialText)

  // Expose update method so callers can sync when textarea changes
  ;(section as unknown as { updateText: (t: string) => void }).updateText = rebuild

  return section
}

function renderEditorEmpty(container: HTMLElement): void {
  container.innerHTML = ''
  const el = document.createElement('div')
  el.className = 'ws-editor-empty'
  el.textContent = t('selectBubbleHint')
  container.appendChild(el)
}

function renderEditor(
  container: HTMLElement,
  bubble: MangaBubble,
  idx: number,
  total: number,
  callbacks: EditorCallbacks,
  computedFontSize?: number,
): void {
  container.innerHTML = ''
  const locked = bubble.is_locked
  const editor = document.createElement('div')
  editor.className = 'ws-editor'

  // JA field
  const jaLabel = document.createElement('div')
  jaLabel.className = 'ws-editor-label'
  jaLabel.textContent = t('jaLabel')

  const jaTextarea = document.createElement('textarea')
  jaTextarea.className = 'ws-editor-textarea'
  jaTextarea.rows = 2
  jaTextarea.value = bubble.raw_ja
  jaTextarea.readOnly = locked
  jaTextarea.addEventListener('input', () => callbacks.onTextChange('raw_ja', jaTextarea.value))

  // ZH field
  const zhLabel = document.createElement('div')
  zhLabel.className = 'ws-editor-label'
  zhLabel.textContent = t('zhLabel')

  const zhTextarea = document.createElement('textarea')
  zhTextarea.className = 'ws-editor-textarea'
  zhTextarea.rows = 3
  zhTextarea.value = bubble.translated_zh
  zhTextarea.readOnly = locked
  zhTextarea.addEventListener('input', () => callbacks.onTextChange('translated_zh', zhTextarea.value))

  // Nav row
  const nav = document.createElement('div')
  nav.className = 'ws-editor-nav'

  const prevBtn = document.createElement('button')
  prevBtn.type = 'button'
  prevBtn.className = 'ws-nav-btn'
  prevBtn.textContent = t('prevBtn')
  prevBtn.disabled = idx === 0
  prevBtn.addEventListener('click', () => callbacks.onNavigate(-1))

  const pos = document.createElement('span')
  pos.className = 'ws-editor-pos'
  pos.textContent = `#${idx + 1} / ${total}`

  const nextBtn = document.createElement('button')
  nextBtn.type = 'button'
  nextBtn.className = 'ws-nav-btn'
  nextBtn.textContent = t('nextBtn')
  nextBtn.disabled = idx === total - 1
  nextBtn.addEventListener('click', () => callbacks.onNavigate(1))

  nav.appendChild(prevBtn)
  nav.appendChild(pos)
  nav.appendChild(nextBtn)

  // Actions row
  const actions = document.createElement('div')
  actions.className = 'ws-editor-actions'

  const badge = makeBadge(bubble.state, 'ws-editor-badge')

  const lockBtn = document.createElement('button')
  lockBtn.type = 'button'
  lockBtn.className = 'ws-lock-btn'
  lockBtn.textContent = locked ? t('unlockBtn') : t('lockBtn')
  lockBtn.addEventListener('click', () => callbacks.onLockToggle())

  actions.appendChild(badge)
  actions.appendChild(lockBtn)

  if (callbacks.onRevertInpaint) {
    const revertBtn = document.createElement('button')
    revertBtn.type = 'button'
    revertBtn.className = 'ws-revert-inpaint-btn'
    revertBtn.textContent = t('revertInpaint')
    revertBtn.addEventListener('click', () => callbacks.onRevertInpaint!())
    actions.appendChild(revertBtn)
  }

  if (callbacks.onRevertTypeset) {
    const revertBtn = document.createElement('button')
    revertBtn.type = 'button'
    revertBtn.className = 'ws-revert-typeset-btn'
    revertBtn.textContent = t('revertTypeset')
    revertBtn.addEventListener('click', () => callbacks.onRevertTypeset!())
    actions.appendChild(revertBtn)
  }

  // Cover section — toggle background fill behind typeset text
  const coverSection = document.createElement('div')
  coverSection.className = 'ws-cover-section'

  const coverLabel = document.createElement('label')
  coverLabel.className = 'ws-cover-label'
  const coverCheck = document.createElement('input')
  coverCheck.type = 'checkbox'
  coverCheck.checked = bubble.cover ?? false
  coverCheck.addEventListener('change', () => callbacks.onCoverChange(coverCheck.checked))
  coverLabel.appendChild(coverCheck)
  coverLabel.append(t('coverBgLabel'))

  const coverControls = document.createElement('div')
  coverControls.className = 'ws-cover-controls'
  coverControls.hidden = !coverCheck.checked

  const coverShapeSelect = document.createElement('select')
  coverShapeSelect.className = 'ws-shape-select'
  coverShapeSelect.innerHTML = `<option value="rect">${t('shapeRect')}</option><option value="bubble">${t('shapeBubble')}</option>`
  coverShapeSelect.value = bubble.shape ?? 'rect'
  coverShapeSelect.addEventListener('change', () =>
    callbacks.onShapeChange(coverShapeSelect.value as 'rect' | 'bubble'))

  const outlineLabel = document.createElement('label')
  outlineLabel.className = 'ws-cover-label'
  const outlineCheck = document.createElement('input')
  outlineCheck.type = 'checkbox'
  outlineCheck.checked = bubble.coverOutline ?? false
  outlineCheck.addEventListener('change', () => callbacks.onOutlineChange(outlineCheck.checked))
  outlineLabel.appendChild(outlineCheck)
  outlineLabel.append(t('outlineLabel'))

  coverCheck.addEventListener('change', () => { coverControls.hidden = !coverCheck.checked })
  coverControls.appendChild(coverShapeSelect)
  coverControls.appendChild(outlineLabel)
  coverSection.appendChild(coverLabel)
  coverSection.appendChild(coverControls)

  // Font size override row
  const fontSizeRow = document.createElement('div')
  fontSizeRow.className = 'ws-font-size-row'

  const fontSizeLabel = document.createElement('span')
  fontSizeLabel.className = 'ws-editor-label'
  fontSizeLabel.textContent = t('fontSizeLabel')

  const fontSizeInput = document.createElement('input')
  fontSizeInput.type = 'number'
  fontSizeInput.className = 'ws-font-size-input'
  fontSizeInput.min = '8'
  fontSizeInput.max = '72'
  const displaySize = bubble.font_size_override ?? computedFontSize
  if (displaySize !== undefined) fontSizeInput.value = String(displaySize)
  fontSizeInput.addEventListener('change', () => {
    const v = parseInt(fontSizeInput.value, 10)
    if (isNaN(v) || fontSizeInput.value.trim() === '') {
      fontSizeInput.value = computedFontSize !== undefined ? String(computedFontSize) : ''
      callbacks.onFontSizeOverrideChange(undefined)
    } else {
      const clamped = Math.max(8, Math.min(72, v))
      fontSizeInput.value = String(clamped)
      callbacks.onFontSizeOverrideChange(clamped)
    }
  })

  const fontSizeClearBtn = document.createElement('button')
  fontSizeClearBtn.type = 'button'
  fontSizeClearBtn.className = 'ws-font-size-clear'
  fontSizeClearBtn.textContent = '×'
  fontSizeClearBtn.title = 'Reset to auto'
  fontSizeClearBtn.addEventListener('click', () => {
    fontSizeInput.value = computedFontSize !== undefined ? String(computedFontSize) : ''
    callbacks.onFontSizeOverrideChange(undefined)
  })

  fontSizeRow.appendChild(fontSizeLabel)
  fontSizeRow.appendChild(fontSizeInput)
  fontSizeRow.appendChild(fontSizeClearBtn)

  // Direction toggle
  const dirLabel = document.createElement('label')
  dirLabel.className = 'ws-cover-label'
  const dirCheck = document.createElement('input')
  dirCheck.type = 'checkbox'
  dirCheck.checked = bubble.text_direction === 'horizontal'
  dirCheck.addEventListener('change', () =>
    callbacks.onDirectionChange(dirCheck.checked ? 'horizontal' : 'vertical'))
  dirLabel.appendChild(dirCheck)
  dirLabel.append(t('horizontalText'))

  // Background text toggle
  const bgLabel = document.createElement('label')
  bgLabel.className = 'ws-cover-label'
  const bgCheck = document.createElement('input')
  bgCheck.type = 'checkbox'
  bgCheck.checked = bubble.is_background === true
  bgCheck.addEventListener('change', () => callbacks.onIsBackgroundChange(bgCheck.checked))
  bgLabel.appendChild(bgCheck)
  bgLabel.append(t('backgroundText'))

  // Text color selector
  const textColorRow = document.createElement('div')
  textColorRow.className = 'ws-font-size-row'
  const textColorLabel = document.createElement('span')
  textColorLabel.className = 'ws-editor-label'
  textColorLabel.textContent = t('textColorLabel')
  const textColorBlack = document.createElement('button')
  textColorBlack.type = 'button'
  textColorBlack.textContent = t('blackBtn')
  textColorBlack.className = 'ws-text-color-btn' + ((!bubble.text_color || bubble.text_color === 'black') ? ' ws-text-color-active' : '')
  const textColorWhite = document.createElement('button')
  textColorWhite.type = 'button'
  textColorWhite.textContent = t('whiteBtn')
  textColorWhite.className = 'ws-text-color-btn' + (bubble.text_color === 'white' ? ' ws-text-color-active' : '')
  textColorBlack.addEventListener('click', () => {
    callbacks.onTextColorChange('black')
    textColorBlack.classList.add('ws-text-color-active')
    textColorWhite.classList.remove('ws-text-color-active')
  })
  textColorWhite.addEventListener('click', () => {
    callbacks.onTextColorChange('white')
    textColorWhite.classList.add('ws-text-color-active')
    textColorBlack.classList.remove('ws-text-color-active')
  })
  textColorRow.appendChild(textColorLabel)
  textColorRow.appendChild(textColorBlack)
  textColorRow.appendChild(textColorWhite)

  // Rotation row
  const rotationRow = document.createElement('div')
  rotationRow.className = 'ws-font-size-row'

  const rotationLabel = document.createElement('span')
  rotationLabel.className = 'ws-editor-label'
  rotationLabel.textContent = t('rotationLabel')

  const rotationInput = document.createElement('input')
  rotationInput.type = 'number'
  rotationInput.className = 'ws-font-size-input'
  rotationInput.min = '-45'
  rotationInput.max = '45'
  if (bubble.rotation !== undefined) rotationInput.value = String(bubble.rotation)
  rotationInput.addEventListener('change', () => {
    const v = parseInt(rotationInput.value, 10)
    if (isNaN(v) || rotationInput.value.trim() === '') {
      rotationInput.value = ''
      callbacks.onRotationChange(undefined)
    } else {
      const clamped = Math.max(-45, Math.min(45, v))
      rotationInput.value = String(clamped)
      callbacks.onRotationChange(clamped)
    }
  })

  const rotationClearBtn = document.createElement('button')
  rotationClearBtn.type = 'button'
  rotationClearBtn.className = 'ws-font-size-clear'
  rotationClearBtn.textContent = '×'
  rotationClearBtn.title = 'Remove rotation'
  rotationClearBtn.addEventListener('click', () => {
    rotationInput.value = ''
    callbacks.onRotationChange(undefined)
  })

  rotationRow.appendChild(rotationLabel)
  rotationRow.appendChild(rotationInput)
  rotationRow.appendChild(rotationClearBtn)

  const dragHint = document.createElement('div')
  dragHint.className = 'ws-drag-hint'
  dragHint.textContent = t('dragHint')

  const resetPosBtn = document.createElement('button')
  resetPosBtn.type = 'button'
  resetPosBtn.className = 'ws-font-size-clear'
  resetPosBtn.textContent = t('resetPosition')
  resetPosBtn.title = 'Clear text offset — return text to default centered position'
  resetPosBtn.addEventListener('click', () => {
    bubble.text_offset_x = 0
    bubble.text_offset_y = 0
    callbacks.onResetPosition()
  })

  // Ruby panel — only shown for vertical text (horizontal doesn't render ruby)
  const rubyPanel = renderRubyPanel(bubble.translated_zh, (newText) => {
    bubble.translated_zh = newText
    zhTextarea.value = newText
    callbacks.onTextChange('translated_zh', newText)
  })
  rubyPanel.style.display = bubble.text_direction === 'horizontal' ? 'none' : ''

  // Keep ruby panel in sync when user edits textarea directly
  zhTextarea.addEventListener('input', () => {
    ;(rubyPanel as unknown as { updateText: (t: string) => void }).updateText(zhTextarea.value)
  })

  editor.appendChild(jaLabel)
  editor.appendChild(jaTextarea)
  editor.appendChild(zhLabel)
  editor.appendChild(zhTextarea)
  editor.appendChild(rubyPanel)
  editor.appendChild(fontSizeRow)
  editor.appendChild(rotationRow)
  editor.appendChild(dirLabel)
  editor.appendChild(bgLabel)
  editor.appendChild(textColorRow)
  editor.appendChild(dragHint)
  editor.appendChild(resetPosBtn)
  editor.appendChild(coverSection)
  editor.appendChild(nav)
  editor.appendChild(actions)
  container.appendChild(editor)

  // Auto-focus ZH textarea when not locked
  if (!locked) zhTextarea.focus()
}

// ── Bubble list ───────────────────────────────────────────────────────────────

function rebuildBubbleList(
  bubbles: MangaBubble[],
  sortedIds: string[],
  listEl: HTMLElement,
  selectFn: (id: string) => void,
  deleteFn: (id: string) => void,
): void {
  listEl.innerHTML = ''
  sortedIds.forEach((id, idx) => {
    const bubble = bubbles.find(b => b.id === id)!

    const item = document.createElement('div')
    item.className = 'ws-bubble-item'
    item.dataset.id = id

    const num = document.createElement('span')
    num.className = 'ws-bubble-num'
    num.textContent = `#${idx + 1}`

    const preview = document.createElement('span')
    preview.className = 'ws-bubble-preview'
    preview.dataset.preview = id
    preview.textContent = truncate(bubble.raw_ja || '(no OCR)', 18)

    const badge = makeBadge(bubble.state)
    badge.dataset.badge = id

    const delBtn = document.createElement('button')
    delBtn.type = 'button'
    delBtn.className = 'ws-delete-btn'
    delBtn.textContent = '×'
    delBtn.title = 'Remove bubble'
    delBtn.addEventListener('click', (e) => {
      e.stopPropagation()
      deleteFn(id)
    })

    item.appendChild(num)
    item.appendChild(preview)
    item.appendChild(badge)
    item.appendChild(delBtn)
    item.addEventListener('click', () => selectFn(id))
    listEl.appendChild(item)
  })
}

// ── SVG overlay ───────────────────────────────────────────────────────────────

// rx/ry in SVG percentage units (viewBox 0 0 100 100)
function overlayRxRy(rect: { w: number; h: number }, shape: 'rect' | 'bubble' | 'freehand' | undefined): number {
  if (shape === 'bubble') return Math.min(rect.w, rect.h) * 0.40
  return 0
}

/** SVG matrix transform that rotates visually by `deg` degrees around (cx, cy) in percentage space,
 *  compensating for the image aspect ratio so the result looks like a true rectangle on screen. */
function overlayRotateTransform(deg: number, cx: number, cy: number, imgW: number, imgH: number): string {
  const θ = deg * Math.PI / 180
  const cos = Math.cos(θ), sin = Math.sin(θ)
  const sx = imgW / 100, sy = imgH / 100   // px per SVG unit
  // Matrix: S^-1 × R × S, where S = diag(sx, sy)
  const a = cos,               c = -sin * sy / sx
  const b = sin * sx / sy,     d = cos
  const e = cx * (1 - a) - c * cy
  const f = cy * (1 - d) - b * cx
  return `matrix(${a},${b},${c},${d},${e},${f})`
}

function rebuildSvgOverlay(
  bubbles: MangaBubble[],
  svg: SVGSVGElement,
  mousedownFn: (id: string, e: MouseEvent) => void,
  imgW: number,
  imgH: number,
): void {
  while (svg.firstChild) svg.removeChild(svg.firstChild)

  bubbles.forEach(bubble => {
    if (bubble.shape === 'freehand' && bubble.points && bubble.points.length >= 3) {
      const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
      poly.setAttribute('points', bubble.points.map(p => `${p.x},${p.y}`).join(' '))
      poly.setAttribute('vector-effect', 'non-scaling-stroke')
      poly.setAttribute('pointer-events', 'fill')
      poly.classList.add('ws-bubble-rect', 'bubble-polygon')
      poly.dataset.id = bubble.id
      poly.addEventListener('mousedown', (e) => { e.preventDefault(); mousedownFn(bubble.id, e) })
      svg.appendChild(poly)
    } else {
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      rect.setAttribute('x', String(bubble.rect.x))
      rect.setAttribute('y', String(bubble.rect.y))
      rect.setAttribute('width', String(bubble.rect.w))
      rect.setAttribute('height', String(bubble.rect.h))
      const r = overlayRxRy(bubble.rect, bubble.shape)
      rect.setAttribute('rx', String(r))
      rect.setAttribute('ry', String(r))
      rect.setAttribute('vector-effect', 'non-scaling-stroke')
      if (bubble.rotation) {
        const cx = bubble.rect.x + bubble.rect.w / 2
        const cy = bubble.rect.y + bubble.rect.h / 2
        rect.setAttribute('transform', overlayRotateTransform(bubble.rotation, cx, cy, imgW, imgH))
      }
      rect.classList.add('ws-bubble-rect')
      rect.dataset.id = bubble.id
      rect.addEventListener('mousedown', (e) => { e.preventDefault(); mousedownFn(bubble.id, e) })
      svg.appendChild(rect)
    }
  })
}

// ── Main entry point ──────────────────────────────────────────────────────────

export function renderWorkspace(container: HTMLElement, page: MangaPage): void {
  let imageUrl = URL.createObjectURL(page.imageBlob)

  // ── Mutable state ──────────────────────────────────────────────────────────
  let bubbles: MangaBubble[] = []
  let sortedIds: string[] = []
  let selectedId: string | null = null
  let pageMask: DetectionMask | null = null
  // Per-bubble inpaint patches: canvas region captured BEFORE each inpaint run
  // putImageData restores exactly those pixels — no need to re-detect
  const inpaintPatches = new Map<string, { data: ImageData; x: number; y: number }>()
  const computedFontSizes: Record<string, number> = {}

  // ── Build DOM ──────────────────────────────────────────────────────────────
  container.innerHTML = ''

  const root = document.createElement('div')
  root.className = 'ws-root'
  container.appendChild(root)

  // Topbar
  const topbar = document.createElement('div')
  topbar.className = 'ws-topbar'
  topbar.innerHTML = `
    <span class="ws-topbar-brand">Kalar</span>
    <span style="flex:1"></span>
    <div class="ws-lang-toggle">
      <button type="button" class="ws-lang-btn" data-lang="en">EN</button>
      <button type="button" class="ws-lang-btn" data-lang="zh">中文</button>
    </div>
    <button type="button" class="ws-settings-btn" data-i18n="settingsBtn"></button>
    <button type="button" class="ws-new-btn" data-i18n="newPageBtn"></button>
  `
  applyLocale(topbar)
  const langBtns = topbar.querySelectorAll<HTMLButtonElement>('.ws-lang-btn')
  function syncLangBtns() {
    langBtns.forEach(btn => btn.classList.toggle('is-active', btn.dataset.lang === getLocale()))
  }
  syncLangBtns()
  langBtns.forEach(btn => btn.addEventListener('click', () => setLocale(btn.dataset.lang as 'en' | 'zh')))
  root.appendChild(topbar)

  topbar.querySelector<HTMLButtonElement>('.ws-settings-btn')!.addEventListener('click', () => {
    openSettings()
  })

  topbar.querySelector<HTMLButtonElement>('.ws-new-btn')!.addEventListener('click', () => {
    ac.abort()
    URL.revokeObjectURL(imageUrl)
    import('./app').then(m => m.renderApp(container))
  })

  // Content row
  const content = document.createElement('div')
  content.className = 'ws-content'
  root.appendChild(content)

  // Declared early — assigned in Add Bubble section below, used in lasso event handlers
  let lassoBtn!: HTMLButtonElement

  // Left: dictionary panel
  const dictPanel = document.createElement('div')
  dictPanel.className = 'ws-dict-panel'
  content.appendChild(dictPanel)
  renderDictPanel(dictPanel)

  // Add Bubble section (bottom of left sidebar)
  const addBubbleSection = document.createElement('div')
  addBubbleSection.className = 'ws-add-bubble-section'

  const addBubbleRow = document.createElement('div')
  addBubbleRow.className = 'ws-add-bubble-row'

  const addBoxBtn = document.createElement('button')
  addBoxBtn.type = 'button'
  addBoxBtn.className = 'ws-add-bubble-btn'
  addBoxBtn.dataset.i18n = 'addBubbleBtn'
  addBoxBtn.textContent = t('addBubbleBtn')

  const addBubbleWrapper = document.createElement('div')
  addBubbleWrapper.className = 'ws-add-bubble-wrapper'

  const modePill = document.createElement('button')
  modePill.type = 'button'
  modePill.className = 'ws-add-bubble-pill'
  modePill.innerHTML = `<span class="ws-pill-label">${t('boxOption')}</span><span class="ws-pill-arrow">▾</span>`

  const addBubbleDropdown = document.createElement('div')
  addBubbleDropdown.className = 'ws-add-bubble-dropdown'
  addBubbleDropdown.hidden = true

  const dropdownItems: { icon: string; iconColor: string; labelKey: I18nKey; textSpan?: HTMLSpanElement; action: () => void }[] = [
    { icon: '▭', iconColor: '#63b3ed', labelKey: 'boxOption',     action: () => { addManualBubble({ x: 40, y: 40, w: 20, h: 20 }, 'rect') } },
    { icon: '◯', iconColor: '#68d391', labelKey: 'roundOption',   action: () => { addManualBubble({ x: 40, y: 40, w: 20, h: 20 }, 'bubble') } },
    { icon: '✏', iconColor: '#f6ad55', labelKey: 'frehandOption', action: () => {
      lassoMode = true
      lassoBtn.classList.add('is-active')
      svg.style.cursor = 'crosshair'
      svg.style.pointerEvents = 'all'
    }},
  ]

  let currentMode = dropdownItems[0]

  for (const item of dropdownItems) {
    const opt = document.createElement('button')
    opt.type = 'button'
    opt.className = 'ws-add-bubble-option'
    const iconSpan = document.createElement('span')
    iconSpan.textContent = item.icon
    iconSpan.style.color = item.iconColor
    iconSpan.style.fontSize = '1rem'
    const textSpan = document.createElement('span')
    textSpan.textContent = t(item.labelKey)
    item.textSpan = textSpan
    opt.appendChild(iconSpan)
    opt.appendChild(textSpan)
    opt.addEventListener('click', () => {
      currentMode = item
      modePill.querySelector<HTMLSpanElement>('.ws-pill-label')!.textContent = t(item.labelKey)
      addBubbleDropdown.hidden = true
    })
    addBubbleDropdown.appendChild(opt)
  }

  addBoxBtn.addEventListener('click', () => { currentMode.action() })

  modePill.addEventListener('click', (e) => {
    e.stopPropagation()
    addBubbleDropdown.hidden = !addBubbleDropdown.hidden
  })

  document.addEventListener('click', () => { addBubbleDropdown.hidden = true })

  // lassoBtn kept as a no-op handle so lasso exit logic still works
  lassoBtn = document.createElement('button')
  lassoBtn.hidden = true

  addBubbleWrapper.appendChild(modePill)
  addBubbleWrapper.appendChild(addBubbleDropdown)
  addBubbleRow.appendChild(addBoxBtn)
  addBubbleRow.appendChild(addBubbleWrapper)
  addBubbleSection.appendChild(addBubbleRow)
  dictPanel.appendChild(addBubbleSection)

  // Centre: image viewer
  const viewer = document.createElement('div')
  viewer.className = 'ws-viewer'
  content.appendChild(viewer)

  const imageFrame = document.createElement('div')
  imageFrame.className = 'ws-image-frame'
  viewer.appendChild(imageFrame)

  const img = document.createElement('img')
  img.src = imageUrl
  img.alt = page.filename
  imageFrame.appendChild(img)

  // Inpaint canvas — transparent layer sitting above the image (never modifies img.src)
  const inpaintCanvas = document.createElement('canvas')
  inpaintCanvas.className = 'ws-inpaint-layer'
  imageFrame.appendChild(inpaintCanvas)

  // Set layer dimensions once the image natural size is known
  const typesetSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  typesetSvg.classList.add('ws-typeset-layer')

  img.addEventListener('load', () => {
    inpaintCanvas.width  = img.naturalWidth
    inpaintCanvas.height = img.naturalHeight
    typesetSvg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`)
  }, { once: true })

  // SVG overlay (viewBox 0 0 100 100 maps directly to percentage coords)
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  svg.setAttribute('viewBox', '0 0 100 100')
  svg.setAttribute('preserveAspectRatio', 'none')
  svg.classList.add('ws-bubble-overlay')
  imageFrame.appendChild(svg)

  // Typeset SVG appended after bubble overlay so text groups sit on top
  // and can receive drag events without the bubble overlay intercepting them.
  // pointer-events: none on the SVG itself; individual bubbleGroups get
  // pointer-events: all in renderTypeset so empty areas still select bubbles.
  imageFrame.appendChild(typesetSvg)

  svg.addEventListener('mousedown', (e) => {
    if (!lassoMode) return
    e.preventDefault()
    startLassoDraw(e)
  })

  // ── Drag / resize ─────────────────────────────────────────────────────────
  interface DragState {
    mode: 'move' | 'resize'
    id: string
    handle: string
    startPct: { x: number; y: number }
    startRect: { x: number; y: number; w: number; h: number }
    startPoints?: { x: number; y: number }[]  // snapshot of freehand points at drag start
  }
  let dragState: DragState | null = null

  type BubbleShape = 'rect' | 'bubble'

  // Lasso draw state
  let lassoMode = false
  let lassoDrawing = false
  let lassoPoints: { x: number; y: number }[] = []
  let lassoPreviewEl: SVGPolylineElement | null = null

  const ac = new AbortController()

  function clientToSvgPct(e: MouseEvent): { x: number; y: number } {
    const r = svg.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100)),
      y: Math.max(0, Math.min(100, (e.clientY - r.top) / r.height * 100)),
    }
  }

  function syncSvgRect(bubble: MangaBubble): void {
    const el = svg.querySelector<SVGElement>(`[data-id="${bubble.id}"]`)
    if (!el) return
    if (bubble.shape === 'freehand' && bubble.points) {
      el.setAttribute('points', bubble.points.map(p => `${p.x},${p.y}`).join(' '))
    } else {
      el.setAttribute('x', String(bubble.rect.x))
      el.setAttribute('y', String(bubble.rect.y))
      el.setAttribute('width', String(bubble.rect.w))
      el.setAttribute('height', String(bubble.rect.h))
      const r = overlayRxRy(bubble.rect, bubble.shape)
      el.setAttribute('rx', String(r))
      el.setAttribute('ry', String(r))
      if (bubble.rotation) {
        const cx = bubble.rect.x + bubble.rect.w / 2
        const cy = bubble.rect.y + bubble.rect.h / 2
        el.setAttribute('transform', overlayRotateTransform(bubble.rotation, cx, cy, img.naturalWidth, img.naturalHeight))
      } else {
        el.removeAttribute('transform')
      }
    }
  }

  function renderHandles(bubble: MangaBubble): void {
    svg.querySelector('.ws-handles-group')?.remove()
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.classList.add('ws-handles-group')
    const { x, y, w, h } = bubble.rect
    const S = 1.2  // handle size in SVG % units (~7px on a 600px display)
    const pts = [
      { id: 'nw', cx: x,       cy: y,       cur: 'nw-resize' },
      { id: 'n',  cx: x + w/2, cy: y,       cur: 'n-resize'  },
      { id: 'ne', cx: x + w,   cy: y,       cur: 'ne-resize' },
      { id: 'e',  cx: x + w,   cy: y + h/2, cur: 'e-resize'  },
      { id: 'se', cx: x + w,   cy: y + h,   cur: 'se-resize' },
      { id: 's',  cx: x + w/2, cy: y + h,   cur: 's-resize'  },
      { id: 'sw', cx: x,       cy: y + h,   cur: 'sw-resize' },
      { id: 'w',  cx: x,       cy: y + h/2, cur: 'w-resize'  },
    ]
    for (const p of pts) {
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
      el.setAttribute('x', String(p.cx - S / 2))
      el.setAttribute('y', String(p.cy - S / 2))
      el.setAttribute('width',  String(S))
      el.setAttribute('height', String(S))
      el.setAttribute('data-handle', p.id)
      el.classList.add('ws-handle-sq')
      el.style.cursor = p.cur
      el.addEventListener('mousedown', (e) => {
        if (lassoMode) return
        e.stopPropagation()
        e.preventDefault()
        dragState = {
          mode: 'resize', id: bubble.id, handle: p.id,
          startPct: clientToSvgPct(e), startRect: { ...bubble.rect },
        }
      })
      g.appendChild(el)
    }
    if (bubble.rotation) {
      const cx = x + w / 2, cy = y + h / 2
      g.setAttribute('transform', overlayRotateTransform(bubble.rotation, cx, cy, img.naturalWidth, img.naturalHeight))
    }
    svg.appendChild(g)
  }

  function clearHandles(): void {
    svg.querySelector('.ws-handles-group')?.remove()
  }

  document.addEventListener('mousemove', (e: MouseEvent) => {
    if (lassoDrawing) { continueLassoDraw(e); return }
    if (!dragState) return
    const cur = clientToSvgPct(e)
    const dx = cur.x - dragState.startPct.x
    const dy = cur.y - dragState.startPct.y
    const bubble = bubbles.find(b => b.id === dragState!.id)
    if (!bubble) return
    const sr = dragState.startRect
    let { x, y, w, h } = sr

    if (dragState.mode === 'move') {
      x = Math.max(0, Math.min(100 - sr.w, sr.x + dx))
      y = Math.max(0, Math.min(100 - sr.h, sr.y + dy))
      // Translate freehand polygon: apply offset relative to startPoints (not current points)
      // so accumulated floating-point drift never occurs.
      if (bubble.shape === 'freehand' && dragState.startPoints) {
        const ptsDx = x - sr.x
        const ptsDy = y - sr.y
        bubble.points = dragState.startPoints.map(p => ({ x: p.x + ptsDx, y: p.y + ptsDy }))
      }
    } else {
      // For rotated bubbles, project drag onto the box's local axes
      let ldx = dx, ldy = dy
      if (bubble.rotation) {
        const θ = bubble.rotation * Math.PI / 180
        const cos = Math.cos(θ), sin = Math.sin(θ)
        const W = img.naturalWidth, H = img.naturalHeight
        ldx = dx * cos + dy * sin * (H / W)
        ldy = -dx * sin * (W / H) + dy * cos
      }
      switch (dragState.handle) {
        case 'nw': x = sr.x + ldx; y = sr.y + ldy; w = sr.w - ldx; h = sr.h - ldy; break
        case 'n':                   y = sr.y + ldy;                  h = sr.h - ldy; break
        case 'ne':                  y = sr.y + ldy; w = sr.w + ldx;  h = sr.h - ldy; break
        case 'e':                                   w = sr.w + ldx;                  break
        case 'se':                                  w = sr.w + ldx;  h = sr.h + ldy; break
        case 's':                                                     h = sr.h + ldy; break
        case 'sw': x = sr.x + ldx;                 w = sr.w - ldx;  h = sr.h + ldy; break
        case 'w':  x = sr.x + ldx;                 w = sr.w - ldx;                  break
      }
      w = Math.max(2, w)
      h = Math.max(2, h)
      x = Math.max(0, Math.min(100 - w, x))
      y = Math.max(0, Math.min(100 - h, y))
    }

    bubble.rect = { x, y, w, h }
    syncSvgRect(bubble)
    renderHandles(bubble)
  }, { signal: ac.signal })

  document.addEventListener('mouseup', () => {
    if (lassoDrawing) { finishLassoDraw(); return }
    dragState = null
  }, { signal: ac.signal })


  // Controls row
  const controls = document.createElement('div')
  controls.className = 'ws-viewer-controls'
  viewer.appendChild(controls)

  const detectBtn = document.createElement('button')
  detectBtn.type = 'button'
  detectBtn.className = 'ws-detect-btn'
  detectBtn.dataset.i18n = 'detectBubbles'
  detectBtn.textContent = t('detectBubbles')
  controls.appendChild(detectBtn)

  const ocrBtn = document.createElement('button')
  ocrBtn.type = 'button'
  ocrBtn.className = 'ws-ocr-btn'
  ocrBtn.dataset.i18n = 'ocrAll'
  ocrBtn.textContent = t('ocrAll')
  ocrBtn.disabled = true
  controls.appendChild(ocrBtn)


  const copyPromptBtn = document.createElement('button')
  copyPromptBtn.type = 'button'
  copyPromptBtn.className = 'ws-copy-prompt-btn'
  copyPromptBtn.dataset.i18n = 'copyPrompt'
  copyPromptBtn.textContent = t('copyPrompt')
  copyPromptBtn.disabled = true
  controls.appendChild(copyPromptBtn)

  const pasteResponseBtn = document.createElement('button')
  pasteResponseBtn.type = 'button'
  pasteResponseBtn.className = 'ws-paste-btn'
  pasteResponseBtn.dataset.i18n = 'pasteResponse'
  pasteResponseBtn.textContent = t('pasteResponse')
  pasteResponseBtn.disabled = true
  controls.appendChild(pasteResponseBtn)

  const inpaintBtn = document.createElement('button')
  inpaintBtn.type = 'button'
  inpaintBtn.className = 'ws-inpaint-btn'
  inpaintBtn.dataset.i18n = 'inpaintAll'
  inpaintBtn.textContent = t('inpaintAll')
  inpaintBtn.disabled = true
  controls.appendChild(inpaintBtn)

  const revertAllBtn = document.createElement('button')
  revertAllBtn.type = 'button'
  revertAllBtn.className = 'ws-revert-all-btn'
  revertAllBtn.dataset.i18n = 'revertAllInpaint'
  revertAllBtn.textContent = t('revertAllInpaint')
  revertAllBtn.title = 'Clear the entire inpaint canvas — restores original pixels everywhere'
  controls.appendChild(revertAllBtn)

  revertAllBtn.addEventListener('click', () => {
    const ctx = inpaintCanvas.getContext('2d')!
    ctx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height)
    inpaintPatches.clear()
    // Re-render editor so "Revert Inpaint" button disappears
    if (selectedId) selectBubble(selectedId)
  })

  const typesetBtn = document.createElement('button')
  typesetBtn.type = 'button'
  typesetBtn.className = 'ws-typeset-btn'
  typesetBtn.dataset.i18n = 'typesetAll'
  typesetBtn.textContent = t('typesetAll')
  typesetBtn.disabled = true
  controls.appendChild(typesetBtn)

  const revertAllTypesetBtn = document.createElement('button')
  revertAllTypesetBtn.type = 'button'
  revertAllTypesetBtn.className = 'ws-revert-all-typeset-btn'
  revertAllTypesetBtn.dataset.i18n = 'revertAllTypeset'
  revertAllTypesetBtn.textContent = t('revertAllTypeset')
  revertAllTypesetBtn.title = 'Clear all typeset text from the SVG layer'
  controls.appendChild(revertAllTypesetBtn)

  revertAllTypesetBtn.addEventListener('click', () => {
    while (typesetSvg.firstChild) typesetSvg.removeChild(typesetSvg.firstChild)
    if (selectedId) selectBubble(selectedId)
  })

  // ── Text drag ─────────────────────────────────────────────────────────────
  let dragBubbleId:  string | null = null
  let dragGroupEl:   SVGGElement | null = null
  let dragStartX = 0
  let dragStartY = 0

  typesetSvg.addEventListener('mousedown', (e) => {
    let el = e.target as Element | null
    while (el && el !== typesetSvg) {
      if (el.getAttribute('data-bubble-id')) break
      el = el.parentElement
    }
    if (!el || el === typesetSvg) return
    dragBubbleId = el.getAttribute('data-bubble-id')
    dragGroupEl  = (el as Element).querySelector<SVGGElement>('[data-text-group]')
    dragStartX   = e.clientX
    dragStartY   = e.clientY
    typesetSvg.style.cursor = 'grabbing'
    svg.style.pointerEvents = 'none'
    e.preventDefault()
  })

  const onTextDragMove = (e: MouseEvent) => {
    if (!dragGroupEl) return
    const svgRect  = typesetSvg.getBoundingClientRect()
    const vb       = typesetSvg.getAttribute('viewBox')!.split(' ').map(Number)
    const scaleX   = vb[2] / svgRect.width
    const scaleY   = vb[3] / svgRect.height
    const dx = (e.clientX - dragStartX) * scaleX
    const dy = (e.clientY - dragStartY) * scaleY
    dragGroupEl.setAttribute('transform', `translate(${dx},${dy})`)
  }

  const onTextDragEnd = (e: MouseEvent) => {
    if (!dragBubbleId || !dragGroupEl) return
    const svgRect  = typesetSvg.getBoundingClientRect()
    const vb       = typesetSvg.getAttribute('viewBox')!.split(' ').map(Number)
    const W = vb[2], H = vb[3]
    const scaleX   = W / svgRect.width
    const scaleY   = H / svgRect.height
    const dx = (e.clientX - dragStartX) * scaleX
    const dy = (e.clientY - dragStartY) * scaleY
    const bubble = bubbles.find(b => b.id === dragBubbleId)
    if (bubble) {
      bubble.text_offset_x = (bubble.text_offset_x ?? 0) + (dx / W) * 100
      bubble.text_offset_y = (bubble.text_offset_y ?? 0) + (dy / H) * 100
      dragGroupEl.removeAttribute('transform')
      const { clippedIds: cids, fontSizes } = renderTypeset(bubbles, typesetSvg)
      Object.assign(computedFontSizes, fontSizes)
      listEl.querySelectorAll('.ws-dot-clip-warn').forEach((el: Element) => el.remove())
      for (const id of cids) {
        const item = listEl.querySelector<HTMLElement>(`[data-id="${id}"]`)
        if (!item) continue
        const warn = document.createElement('span')
        warn.className = 'ws-dot-clip-warn'
        warn.title = 'Some dots were clipped'
        warn.textContent = '⚠ dots clipped'
        item.appendChild(warn)
      }
    }
    dragBubbleId = null
    dragGroupEl  = null
    typesetSvg.style.cursor = ''
    svg.style.pointerEvents = ''
    if (selectedId) selectBubble(selectedId)
  }

  document.addEventListener('mousemove', onTextDragMove)
  document.addEventListener('mouseup',   onTextDragEnd)

  const previewBtn = document.createElement('button')
  previewBtn.type = 'button'
  previewBtn.className = 'ws-preview-btn'
  previewBtn.dataset.i18n = 'preview'
  previewBtn.textContent = t('preview')
  previewBtn.title = 'Hide selection overlays to preview typeset output'
  controls.appendChild(previewBtn)

  previewBtn.addEventListener('click', () => {
    const on = root.classList.toggle('ws-preview-mode')
    previewBtn.dataset.i18n = on ? 'exitPreview' : 'preview'
    previewBtn.textContent = on ? t('exitPreview') : t('preview')
    previewBtn.classList.toggle('is-active', on)
  })

  const downloadBtn = document.createElement('button')
  downloadBtn.type = 'button'
  downloadBtn.className = 'ws-download-btn'
  downloadBtn.dataset.i18n = 'download'
  downloadBtn.textContent = t('download')
  downloadBtn.title = 'Download composited image (original + inpaint layer)'
  controls.appendChild(downloadBtn)

  downloadBtn.addEventListener('click', () => {
    downloadBtn.disabled = true
    try {
      // Composite: original → inpaint overlay → typeset SVG
      const W = img.naturalWidth
      const H = img.naturalHeight
      const canvas = document.createElement('canvas')
      canvas.width  = W
      canvas.height = H
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0)
      ctx.drawImage(inpaintCanvas, 0, 0)

      // Render typeset text directly onto canvas — preserves page-loaded fonts
      if (typesetSvg.childElementCount > 0) {
        renderTypesetToCanvas(bubbles, ctx, W, H)
      }

      canvas.toBlob(blob => {
        if (!blob) return
        const url  = URL.createObjectURL(blob)
        const a    = document.createElement('a')
        const base = page.filename.replace(/\.[^.]+$/, '')
        a.href     = url
        a.download = `${base}_translated.png`
        a.click()
        URL.revokeObjectURL(url)
        downloadBtn.disabled = false
      }, 'image/png')
    } catch {
      downloadBtn.disabled = false
    }
  })

  const statusEl = document.createElement('span')
  statusEl.className = 'ws-detect-status'
  controls.appendChild(statusEl)

  // Right: bubble panel
  const panel = document.createElement('div')
  panel.className = 'ws-panel'
  content.appendChild(panel)

  const panelHeader = document.createElement('div')
  panelHeader.className = 'ws-panel-header'
  panelHeader.innerHTML = `<span class="ws-panel-title" data-i18n="bubblesTitle">${t('bubblesTitle')}</span>`
  const countEl = document.createElement('span')
  countEl.className = 'ws-panel-count'
  countEl.textContent = '0'
  panelHeader.appendChild(countEl)
  panel.appendChild(panelHeader)

  const listEl = document.createElement('div')
  listEl.className = 'ws-bubble-list'
  panel.appendChild(listEl)

  const editorContainer = document.createElement('div')
  panel.appendChild(editorContainer)
  renderEditorEmpty(editorContainer)

  // ── Selection logic ────────────────────────────────────────────────────────

  function selectBubble(id: string): void {
    const prev = selectedId
    selectedId = id

    if (prev) {
      updateSvgSelection(svg, prev, false)
      updateListSelection(listEl, prev, false)
    }
    updateSvgSelection(svg, id, true)
    updateListSelection(listEl, id, true)

    listEl.querySelector<HTMLElement>(`[data-id="${id}"]`)
      ?.scrollIntoView({ block: 'nearest' })

    const bubble = bubbles.find(b => b.id === id)!
    renderHandles(bubble)
    const idx = sortedIds.indexOf(id)

    renderEditor(editorContainer, bubble, idx, sortedIds.length, {
      onTextChange(field, value) {
        bubble[field] = value
        if (field === 'raw_ja') updateListPreview(listEl, id, value)
        if (field === 'translated_zh') {
          updateTypesetBtn()
          if (bubble.state === 'detected' || bubble.state === 'ocr_done') {
            bubble.state = 'translated'
            updateListBadge(listEl, id, bubble.state)
            updateEditorBadge(editorContainer, bubble.state)
          }
        }
      },
      onLockToggle() {
        if (bubble.is_locked) {
          bubble.is_locked = false
          bubble.state = 'translated'
        } else {
          bubble.is_locked = true
          bubble.state = 'reviewed'
        }
        updateListBadge(listEl, id, bubble.state)
        // Re-render editor so textareas get readonly toggled
        selectBubble(id)
      },
      onNavigate(dir) {
        const next = sortedIds[sortedIds.indexOf(id) + dir]
        if (next) selectBubble(next)
      },
      onCoverChange(cover) { bubble.cover = cover },
      onOutlineChange(outline) { bubble.coverOutline = outline },
      onShapeChange(shape) {
        bubble.shape = shape
        const el = svg.querySelector<SVGRectElement>(`[data-id="${id}"]`)
        if (el) {
          const r = overlayRxRy(bubble.rect, shape)
          el.setAttribute('rx', String(r))
          el.setAttribute('ry', String(r))
        }
      },
      onRevertInpaint: inpaintPatches.has(id) ? () => revertBubbleInpaint(id) : undefined,
      onRevertTypeset: typesetSvg.querySelector(`[data-bubble-id="${id}"]`)
        ? () => { typesetSvg.querySelector(`[data-bubble-id="${id}"]`)?.remove(); selectBubble(id) }
        : undefined,
      onFontSizeOverrideChange(size) { bubble.font_size_override = size },
      onDirectionChange(dir) { bubble.text_direction = dir },
      onResetPosition() { selectBubble(id) },
      onIsBackgroundChange(val) { bubble.is_background = val },
      onTextColorChange(color) { bubble.text_color = color },
      onRotationChange(deg) { bubble.rotation = deg; syncSvgRect(bubble) },
    }, computedFontSizes[id])
  }

  // ── Delete bubble ──────────────────────────────────────────────────────────

  function deleteBubble(id: string): void {
    const deletedIdx = sortedIds.indexOf(id)
    bubbles = bubbles.filter(b => b.id !== id)
    sortedIds = sortedIds.filter(sid => sid !== id)

    // Remove from SVG overlay
    svg.querySelector(`[data-id="${id}"]`)?.remove()

    // Remove from list; renumber remaining items
    listEl.querySelector<HTMLElement>(`[data-id="${id}"]`)?.remove()
    listEl.querySelectorAll<HTMLElement>('.ws-bubble-num').forEach((el, i) => {
      el.textContent = `#${i + 1}`
    })

    countEl.textContent = String(bubbles.length)

    // If the deleted bubble was selected, move selection or clear editor
    if (selectedId === id) {
      selectedId = null
      const nextId = sortedIds[deletedIdx] ?? sortedIds[deletedIdx - 1]
      if (nextId) {
        selectBubble(nextId)
      } else {
        clearHandles()
        renderEditorEmpty(editorContainer)
      }
    }

    ocrBtn.disabled = bubbles.length === 0
    inpaintBtn.disabled = bubbles.length === 0
    updateTranslateBtn()
  }

  // ── Inpaint revert ─────────────────────────────────────────────────────

  function revertBubbleInpaint(id: string): void {
    const entry = inpaintPatches.get(id)
    if (!entry) return
    const ctx = inpaintCanvas.getContext('2d')!
    ctx.putImageData(entry.data, entry.x, entry.y)
  }

  // ── Lasso draw functions ───────────────────────────────────────────────

  function startLassoDraw(e: MouseEvent): void {
    lassoDrawing = true
    lassoPoints = [clientToSvgPct(e)]
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline')
    poly.classList.add('lasso-preview')
    svg.appendChild(poly)
    lassoPreviewEl = poly
  }

  function continueLassoDraw(e: MouseEvent): void {
    if (!lassoDrawing) return
    const pt = clientToSvgPct(e)
    const last = lassoPoints[lassoPoints.length - 1]
    const dist = Math.sqrt((pt.x - last.x) ** 2 + (pt.y - last.y) ** 2)
    if (dist < 3) return  // distance filter: keep path under ~100 points
    lassoPoints.push(pt)
    if (lassoPreviewEl) {
      lassoPreviewEl.setAttribute('points', lassoPoints.map(p => `${p.x},${p.y}`).join(' '))
    }
  }

  function finishLassoDraw(): void {
    lassoDrawing = false
    lassoPreviewEl?.remove()
    lassoPreviewEl = null

    const pts = lassoPoints
    lassoPoints = []

    if (pts.length < 3) return  // not enough points

    const xs = pts.map(p => p.x), ys = pts.map(p => p.y)
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    if (maxX - minX < 2 || maxY - minY < 2) return  // bounding box too small

    const rect = { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
    const bubble: MangaBubble = {
      id: crypto.randomUUID(),
      rect,
      points: pts,
      raw_ja: '',
      translated_zh: '',
      state: 'detected',
      is_locked: false,
      layer_z: 0,
      source: 'manual',
      shape: 'freehand',
      cover: true,
    }
    bubbles.push(bubble)
    sortedIds = sortBubbleIds(bubbles)

    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon')
    poly.setAttribute('points', pts.map(p => `${p.x},${p.y}`).join(' '))
    poly.setAttribute('vector-effect', 'non-scaling-stroke')
    poly.setAttribute('pointer-events', 'fill')
    poly.classList.add('ws-bubble-rect', 'bubble-polygon')
    poly.dataset.id = bubble.id
    poly.addEventListener('mousedown', (ev) => {
      if (lassoMode) return
      ev.preventDefault()
      selectBubble(bubble.id)
      dragState = {
        mode: 'move', id: bubble.id, handle: '',
        startPct: clientToSvgPct(ev),
        startRect: { ...bubble.rect },
        startPoints: bubble.points ? [...bubble.points] : undefined,
      }
    })
    svg.appendChild(poly)

    rebuildBubbleList(bubbles, sortedIds, listEl, selectBubble, deleteBubble)
    countEl.textContent = String(bubbles.length)
    ocrBtn.disabled = false
    inpaintBtn.disabled = false
    selectBubble(bubble.id)

    // Exit lasso mode after drawing
    lassoMode = false
    lassoBtn.classList.remove('active')
    svg.style.cursor = ''
    svg.style.pointerEvents = ''
  }

  // ── Manual box creation ────────────────────────────────────────────────

  function addManualBubble(rect: { x: number; y: number; w: number; h: number }, shape: BubbleShape = 'rect'): void {
    const bubble: MangaBubble = {
      id: crypto.randomUUID(),
      rect,
      raw_ja: '',
      translated_zh: '',
      state: 'detected',
      is_locked: false,
      layer_z: 0,
      source: 'manual',
      shape,
      cover: true,
    }
    bubbles.push(bubble)
    sortedIds = sortBubbleIds(bubbles)

    const svgRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    svgRect.setAttribute('x', String(bubble.rect.x))
    svgRect.setAttribute('y', String(bubble.rect.y))
    svgRect.setAttribute('width', String(bubble.rect.w))
    svgRect.setAttribute('height', String(bubble.rect.h))
    const r0 = overlayRxRy(bubble.rect, bubble.shape)
    svgRect.setAttribute('rx', String(r0))
    svgRect.setAttribute('ry', String(r0))
    svgRect.setAttribute('vector-effect', 'non-scaling-stroke')
    svgRect.classList.add('ws-bubble-rect')
    svgRect.dataset.id = bubble.id
    svgRect.addEventListener('mousedown', (e) => {
      if (lassoMode) return
      e.preventDefault()
      selectBubble(bubble.id)
      dragState = {
        mode: 'move', id: bubble.id, handle: '',
        startPct: clientToSvgPct(e),
        startRect: { ...bubble.rect },
      }
    })
    svg.appendChild(svgRect)

    rebuildBubbleList(bubbles, sortedIds, listEl, selectBubble, deleteBubble)
    countEl.textContent = String(bubbles.length)
    ocrBtn.disabled = false
    inpaintBtn.disabled = false
    selectBubble(bubble.id)
  }

  // ── Translate button state helper ─────────────────────────────────────────

  function updateTranslateBtn(): void {
    const hasOcr = bubbles.some(b => b.raw_ja.length > 0)
    copyPromptBtn.disabled = !hasOcr
    pasteResponseBtn.disabled = !hasOcr
  }

  function updateTypesetBtn(): void {
    typesetBtn.disabled = !bubbles.some(b => b.translated_zh.trim().length > 0)
  }

  // ── Detect button ──────────────────────────────────────────────────────────

  detectBtn.addEventListener('click', async () => {
    detectBtn.disabled = true
    statusEl.textContent = 'Starting…'

    try {
      const detected = await detectBubblesWithMask(
        page.imageBlob,
        (stage, _value) => { statusEl.textContent = stage },
      )
      bubbles = detected.bubbles
      pageMask = detected.mask
      sortedIds = sortBubbleIds(bubbles)
      selectedId = null

      // Write sorted debug JSON — entries ordered by panel number so bubble_no matches the UI.
      fetch('/__debug/detect_sorted', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sortedIds.map((id, i) => {
          const b = bubbles.find(b => b.id === id)!
          return {
            bubble_no:        i + 1,
            conf:             b.det_conf,
            mask_density:     b.det_mask_density,
            rect_pct:         b.rect,
          }
        })),
      }).catch(() => {})

      countEl.textContent = String(bubbles.length)
      statusEl.textContent = `${bubbles.length} text regions found`

      rebuildSvgOverlay(bubbles, svg, (id, e) => {
        if (lassoMode) return
        selectBubble(id)
        const b = bubbles.find(b => b.id === id)!
        dragState = {
          mode: 'move', id, handle: '',
          startPct: clientToSvgPct(e),
          startRect: { ...b.rect },
          startPoints: b.points ? [...b.points] : undefined,
        }
      }, img.naturalWidth, img.naturalHeight)
      rebuildBubbleList(bubbles, sortedIds, listEl, selectBubble, deleteBubble)

      // Enable OCR and Inpaint once we have bubbles
      ocrBtn.disabled = bubbles.length === 0
      inpaintBtn.disabled = bubbles.length === 0

      // Auto-select first bubble
      if (sortedIds.length > 0) selectBubble(sortedIds[0])
    } catch (err) {
      statusEl.textContent = 'Detection failed — check console'
      detectBtn.disabled = false
      console.error(err)
    }
  })

  // ── OCR All button ─────────────────────────────────────────────────────────

  ocrBtn.addEventListener('click', async () => {
    ocrBtn.disabled = true
    detectBtn.disabled = true

    const total = bubbles.length
    let done = 0

    for (const id of sortedIds) {
      const bubble = bubbles.find(b => b.id === id)!
      statusEl.textContent = `OCR ${done + 1}/${total}…`

      try {
        const text = await runOCR(
          bubble,
          page.imageBlob,
          (stage, _value) => { statusEl.textContent = `OCR ${done + 1}/${total}: ${stage}` },
        )
        bubble.raw_ja = text
        if (bubble.state === 'detected') bubble.state = 'ocr_done'
        updateListPreview(listEl, id, text)
        updateListBadge(listEl, id, bubble.state)
        if (selectedId === id) {
          updateEditorBadge(editorContainer, bubble.state)
          // Re-render editor to show new raw_ja text
          const idx = sortedIds.indexOf(id)
          renderEditor(editorContainer, bubble, idx, sortedIds.length, {
            onTextChange(field, value) {
              bubble[field] = value
              if (field === 'raw_ja') updateListPreview(listEl, id, value)
              if (field === 'translated_zh') {
                updateTypesetBtn()
                if (bubble.state === 'detected' || bubble.state === 'ocr_done') {
                  bubble.state = 'translated'
                  updateListBadge(listEl, id, bubble.state)
                  updateEditorBadge(editorContainer, bubble.state)
                }
              }
            },
            onLockToggle() {
              if (bubble.is_locked) {
                bubble.is_locked = false
                bubble.state = 'translated'
              } else {
                bubble.is_locked = true
                bubble.state = 'reviewed'
              }
              updateListBadge(listEl, id, bubble.state)
              selectBubble(id)
            },
            onNavigate(dir) {
              const next = sortedIds[sortedIds.indexOf(id) + dir]
              if (next) selectBubble(next)
            },
            onCoverChange(cover) { bubble.cover = cover },
            onOutlineChange(outline) { bubble.coverOutline = outline },
            onShapeChange(shape) {
              bubble.shape = shape
              const el = svg.querySelector<SVGRectElement>(`[data-id="${id}"]`)
              if (el) {
                const r = overlayRxRy(bubble.rect, shape)
                el.setAttribute('rx', String(r))
                el.setAttribute('ry', String(r))
              }
            },
            onRevertInpaint: inpaintPatches.has(id) ? () => revertBubbleInpaint(id) : undefined,
            onRevertTypeset: typesetSvg.querySelector(`[data-bubble-id="${id}"]`)
              ? () => { typesetSvg.querySelector(`[data-bubble-id="${id}"]`)?.remove(); selectBubble(id) }
              : undefined,
            onFontSizeOverrideChange(size) { bubble.font_size_override = size },
            onDirectionChange(dir) { bubble.text_direction = dir },
            onResetPosition() { selectBubble(id) },
            onIsBackgroundChange(val) { bubble.is_background = val },
            onTextColorChange(color) { bubble.text_color = color },
      onRotationChange(deg) { bubble.rotation = deg; syncSvgRect(bubble) },
                }, computedFontSizes[id])
        }
      } catch (err) {
        console.error(`OCR failed for bubble ${id}:`, err)
        statusEl.textContent = `OCR error: ${String(err).slice(0, 80)}`
        ocrBtn.disabled = false
        detectBtn.disabled = false
        return
      }

      done++
    }

    statusEl.textContent = `OCR complete — ${total} bubbles`
    ocrBtn.disabled = false
    detectBtn.disabled = false
    updateTranslateBtn()
  })

  // ── Translate All button ───────────────────────────────────────────────────


  // ── Copy Prompt button ─────────────────────────────────────────────────────

  copyPromptBtn.addEventListener('click', async () => {
    const ocrBubbles = bubbles.filter(b => b.raw_ja.length > 0)
    try {
      await navigator.clipboard.writeText(buildPrompt(ocrBubbles, getGlossary()))
      statusEl.textContent = 'Prompt copied — paste it in your AI chat, then click "Paste Response"'
    } catch {
      statusEl.textContent = 'Clipboard write failed — check browser permissions'
    }
  })

  // ── Paste Response button ──────────────────────────────────────────────────

  pasteResponseBtn.addEventListener('click', async () => {
    try {
      const text = await navigator.clipboard.readText()
      const results = parseTranslationResponse(text)
      for (const { id, translated_zh } of results) {
        const bubble = bubbles.find(b => b.id === id)
        if (!bubble) continue
        bubble.translated_zh = translated_zh
        bubble.state = 'translated'
        updateListBadge(listEl, id, bubble.state)
        if (selectedId === id) selectBubble(id)
      }
      statusEl.textContent = `Applied ${results.length} translations`
      updateTypesetBtn()
    } catch (err) {
      statusEl.textContent = `Paste error: ${String(err).slice(0, 80)}`
    }
  })

  // ── Inpaint All button ─────────────────────────────────────────────────────

  inpaintBtn.addEventListener('click', async () => {
    inpaintBtn.disabled = true
    detectBtn.disabled = true
    ocrBtn.disabled = true
    statusEl.textContent = 'Loading inpaint model (first run: ~208 MB)…'

    try {
      const { blob: resultBlob, expandedRects } = await inpaintPage(
        bubbles,
        page.imageBlob,
        (stage, current, total) => {
          statusEl.textContent = `${stage} (${current}/${total})`
        },
        pageMask,
      )

      // Write expanded bubble interior rects back into each speech bubble
      for (const { id, rect, fillColor } of expandedRects) {
        const b = bubbles.find(b => b.id === id)
        // Don't cache white — white bubbles should always re-route through the white path,
        // not be forced to the solid route via inpaint_color on subsequent runs.
        if (b) { b.bubble_rect = rect; if (fillColor && fillColor !== '#ffffff' && b.shape !== 'bubble' && b.shape !== 'freehand') b.inpaint_color = fillColor }
      }

      // resultBlob is a transparent PNG overlay — speech bubble text rects are white,
      // background text regions have LaMa-reconstructed pixels; rest is alpha=0.
      // Clear first (handles re-runs), then stamp the whole blob at once.
      // The original image (img.src / page.imageBlob) is never modified.
      const inpaintBitmap = await createImageBitmap(resultBlob)
      const ctx = inpaintCanvas.getContext('2d')!
      const W = inpaintCanvas.width
      const H = inpaintCanvas.height

      // Save per-bubble patches BEFORE clearing so each bubble can be individually reverted
      for (const bubble of bubbles) {
        const bx = Math.floor(bubble.rect.x / 100 * W)
        const by = Math.floor(bubble.rect.y / 100 * H)
        const bw = Math.max(1, Math.ceil(bubble.rect.w / 100 * W))
        const bh = Math.max(1, Math.ceil(bubble.rect.h / 100 * H))
        inpaintPatches.set(bubble.id, { data: ctx.getImageData(bx, by, bw, bh), x: bx, y: by })
      }

      ctx.clearRect(0, 0, W, H)
      ctx.drawImage(inpaintBitmap, 0, 0, W, H)
      inpaintBitmap.close()

      // Re-render editor so "Revert Inpaint" button appears for the selected bubble
      if (selectedId) selectBubble(selectedId)

      statusEl.textContent = `Inpainting complete — ${bubbles.length} bubbles erased`
    } catch (err) {
      statusEl.textContent = `Inpaint error: ${String(err).slice(0, 80)}`
      console.error(err)
    }

    inpaintBtn.disabled = false
    detectBtn.disabled = false
    ocrBtn.disabled = bubbles.length === 0
  })

  // ── Typeset All button ─────────────────────────────────────────────────────

  typesetBtn.addEventListener('click', () => {
    const { clippedIds, fontSizes } = renderTypeset(bubbles, typesetSvg)
    Object.assign(computedFontSizes, fontSizes)
    statusEl.textContent = `Typeset ${bubbles.filter(b => b.translated_zh.trim()).length} bubbles`

    // Clear previous dot-clip warnings, then re-add for newly clipped bubbles
    listEl.querySelectorAll('.ws-dot-clip-warn').forEach((el: Element) => el.remove())
    for (const id of clippedIds) {
      const item = listEl.querySelector<HTMLElement>(`[data-id="${id}"]`)
      if (!item) continue
      const warn = document.createElement('span')
      warn.className = 'ws-dot-clip-warn'
      warn.title = 'Some dots were clipped — too many to fit in bubble'
      warn.textContent = '⚠ dots clipped'
      item.appendChild(warn)
    }

    // Re-render editor so "Revert Typeset" button appears for the selected bubble
    if (selectedId) selectBubble(selectedId)
  })

  // ── Locale change handler ──────────────────────────────────────────────────

  onLocaleChange(() => {
    syncLangBtns()
    // Update all data-i18n elements in the workspace
    applyLocale(root)
    // Preview button has dynamic text (not covered by data-i18n when active)
    const isPreview = root.classList.contains('ws-preview-mode')
    previewBtn.dataset.i18n = isPreview ? 'exitPreview' : 'preview'
    previewBtn.textContent = isPreview ? t('exitPreview') : t('preview')
    // Update dropdown option labels and mode pill
    for (const item of dropdownItems) {
      if (item.textSpan) item.textSpan.textContent = t(item.labelKey)
    }
    modePill.querySelector<HTMLSpanElement>('.ws-pill-label')!.textContent = t(currentMode.labelKey)
    // Rebuild bubble list to refresh badge text
    rebuildBubbleList(bubbles, sortedIds, listEl, selectBubble, deleteBubble)
    // Re-render editor panel
    if (selectedId) selectBubble(selectedId)
    else renderEditorEmpty(editorContainer)
    // Re-render dict panel (clears and rebuilds with current locale)
    renderDictPanel(dictPanel)
    dictPanel.appendChild(addBubbleSection)
  })
}
