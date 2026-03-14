import './workspace.css'
import type { MangaBubble, MangaPage, BubbleState } from '../types'
import { detectBubbles } from '../pipeline/detect'
import { runOCR } from '../pipeline/ocr'
import { loadAPIConfig, buildPrompt, translatePage, parseTranslationResponse } from '../pipeline/translate'
import { inpaintPage } from '../pipeline/inpaint'
import { renderTypeset } from '../pipeline/typeset'
import { openSettings } from './settings'

// ── Constants ────────────────────────────────────────────────────────────────

const ROW_THRESHOLD = 5 // bubble y-values within 5% are considered the same row

const STATE_LABELS: Record<BubbleState, string> = {
  detected:   'detected',
  ocr_done:   'ocr done',
  translated: 'translated',
  reviewed:   'reviewed',
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function escapeHtml(str: string): string {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function truncate(str: string, max: number): string {
  return str.length <= max ? str : str.slice(0, max) + '…'
}

function makeBadge(state: BubbleState, extraClass = ''): HTMLSpanElement {
  const el = document.createElement('span')
  el.className = `ws-badge ws-badge--${state}${extraClass ? ' ' + extraClass : ''}`
  el.textContent = STATE_LABELS[state]
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
  el.textContent = STATE_LABELS[state]
}

function updateEditorBadge(editor: HTMLElement, state: BubbleState): void {
  const el = editor.querySelector<HTMLElement>('.ws-editor-badge')
  if (!el) return
  el.className = `ws-badge ws-badge--${state} ws-editor-badge`
  el.textContent = STATE_LABELS[state]
}

// ── Editor rendering ──────────────────────────────────────────────────────────

interface EditorCallbacks {
  onTextChange: (field: 'raw_ja' | 'translated_zh', value: string) => void
  onLockToggle: () => void
  onNavigate: (direction: -1 | 1) => void
}

function renderEditorEmpty(container: HTMLElement): void {
  container.innerHTML = ''
  const el = document.createElement('div')
  el.className = 'ws-editor-empty'
  el.textContent = 'Select a bubble to edit'
  container.appendChild(el)
}

function renderEditor(
  container: HTMLElement,
  bubble: MangaBubble,
  idx: number,
  total: number,
  callbacks: EditorCallbacks,
): void {
  container.innerHTML = ''
  const locked = bubble.is_locked
  const editor = document.createElement('div')
  editor.className = 'ws-editor'

  // JA field
  const jaLabel = document.createElement('div')
  jaLabel.className = 'ws-editor-label'
  jaLabel.textContent = 'Japanese (raw_ja)'

  const jaTextarea = document.createElement('textarea')
  jaTextarea.className = 'ws-editor-textarea'
  jaTextarea.rows = 2
  jaTextarea.value = bubble.raw_ja
  jaTextarea.readOnly = locked
  jaTextarea.addEventListener('input', () => callbacks.onTextChange('raw_ja', jaTextarea.value))

  // ZH field
  const zhLabel = document.createElement('div')
  zhLabel.className = 'ws-editor-label'
  zhLabel.textContent = 'Translation (translated_zh)'

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
  prevBtn.textContent = '← Prev'
  prevBtn.disabled = idx === 0
  prevBtn.addEventListener('click', () => callbacks.onNavigate(-1))

  const pos = document.createElement('span')
  pos.className = 'ws-editor-pos'
  pos.textContent = `#${idx + 1} / ${total}`

  const nextBtn = document.createElement('button')
  nextBtn.type = 'button'
  nextBtn.className = 'ws-nav-btn'
  nextBtn.textContent = 'Next →'
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
  lockBtn.textContent = locked ? 'Unlock' : 'Lock'
  lockBtn.addEventListener('click', () => callbacks.onLockToggle())

  actions.appendChild(badge)
  actions.appendChild(lockBtn)

  editor.appendChild(jaLabel)
  editor.appendChild(jaTextarea)
  editor.appendChild(zhLabel)
  editor.appendChild(zhTextarea)
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

function rebuildSvgOverlay(
  bubbles: MangaBubble[],
  svg: SVGSVGElement,
  mousedownFn: (id: string, e: MouseEvent) => void,
): void {
  while (svg.firstChild) svg.removeChild(svg.firstChild)

  bubbles.forEach(bubble => {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect')
    rect.setAttribute('x', String(bubble.rect.x))
    rect.setAttribute('y', String(bubble.rect.y))
    rect.setAttribute('width', String(bubble.rect.w))
    rect.setAttribute('height', String(bubble.rect.h))
    rect.setAttribute('vector-effect', 'non-scaling-stroke')
    rect.classList.add('ws-bubble-rect')
    rect.dataset.id = bubble.id
    rect.addEventListener('mousedown', (e) => { e.preventDefault(); mousedownFn(bubble.id, e) })
    svg.appendChild(rect)
  })
}

// ── Main entry point ──────────────────────────────────────────────────────────

export function renderWorkspace(container: HTMLElement, page: MangaPage): void {
  let imageUrl = URL.createObjectURL(page.imageBlob)

  // ── Mutable state ──────────────────────────────────────────────────────────
  let bubbles: MangaBubble[] = []
  let sortedIds: string[] = []
  let selectedId: string | null = null

  // ── Build DOM ──────────────────────────────────────────────────────────────
  container.innerHTML = ''

  const root = document.createElement('div')
  root.className = 'ws-root'
  container.appendChild(root)

  // Topbar
  const topbar = document.createElement('div')
  topbar.className = 'ws-topbar'
  topbar.innerHTML = `
    <span class="ws-topbar-brand">MangaVibe</span>
    <span class="ws-topbar-filename">${escapeHtml(page.filename)}</span>
    <button type="button" class="ws-settings-btn">⚙ Settings</button>
    <button type="button" class="ws-new-btn">← New Page</button>
  `
  root.appendChild(topbar)

  topbar.querySelector<HTMLButtonElement>('.ws-settings-btn')!.addEventListener('click', () => {
    openSettings(() => updateTranslateBtn())
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

  // Left: image viewer
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

  // Typeset SVG — Chinese text rendered above the inpaint layer
  // viewBox uses natural image dimensions so font metrics are not distorted
  const typesetSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
  typesetSvg.classList.add('ws-typeset-layer')
  imageFrame.appendChild(typesetSvg)

  // Set layer dimensions once the image natural size is known
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

  // ── Drag / resize ─────────────────────────────────────────────────────────
  interface DragState {
    mode: 'move' | 'resize'
    id: string
    handle: string
    startPct: { x: number; y: number }
    startRect: { x: number; y: number; w: number; h: number }
  }
  let dragState: DragState | null = null

  const ac = new AbortController()

  function clientToSvgPct(e: MouseEvent): { x: number; y: number } {
    const r = svg.getBoundingClientRect()
    return {
      x: Math.max(0, Math.min(100, (e.clientX - r.left) / r.width * 100)),
      y: Math.max(0, Math.min(100, (e.clientY - r.top) / r.height * 100)),
    }
  }

  function syncSvgRect(bubble: MangaBubble): void {
    const el = svg.querySelector<SVGRectElement>(`[data-id="${bubble.id}"]`)
    if (!el) return
    el.setAttribute('x', String(bubble.rect.x))
    el.setAttribute('y', String(bubble.rect.y))
    el.setAttribute('width', String(bubble.rect.w))
    el.setAttribute('height', String(bubble.rect.h))
  }

  function renderHandles(bubble: MangaBubble): void {
    svg.querySelector('.ws-handles-group')?.remove()
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g')
    g.classList.add('ws-handles-group')
    const { x, y, w, h } = bubble.rect
    const pts = [
      { id: 'nw', cx: x,       cy: y,       cur: 'nw-resize', arrow: '↖' },
      { id: 'n',  cx: x + w/2, cy: y,       cur: 'n-resize',  arrow: '↑' },
      { id: 'ne', cx: x + w,   cy: y,       cur: 'ne-resize', arrow: '↗' },
      { id: 'e',  cx: x + w,   cy: y + h/2, cur: 'e-resize',  arrow: '→' },
      { id: 'se', cx: x + w,   cy: y + h,   cur: 'se-resize', arrow: '↘' },
      { id: 's',  cx: x + w/2, cy: y + h,   cur: 's-resize',  arrow: '↓' },
      { id: 'sw', cx: x,       cy: y + h,   cur: 'sw-resize', arrow: '↙' },
      { id: 'w',  cx: x,       cy: y + h/2, cur: 'w-resize',  arrow: '←' },
    ]
    for (const p of pts) {
      const el = document.createElementNS('http://www.w3.org/2000/svg', 'text')
      el.setAttribute('x', String(p.cx))
      el.setAttribute('y', String(p.cy))
      el.setAttribute('text-anchor', 'middle')
      el.setAttribute('dominant-baseline', 'central')
      el.setAttribute('font-size', '2.5')
      el.setAttribute('data-handle', p.id)
      el.classList.add('ws-handle-arrow')
      el.style.cursor = p.cur
      el.textContent = p.arrow
      el.addEventListener('mousedown', (e) => {
        e.stopPropagation()
        e.preventDefault()
        dragState = {
          mode: 'resize', id: bubble.id, handle: p.id,
          startPct: clientToSvgPct(e), startRect: { ...bubble.rect },
        }
      })
      g.appendChild(el)
    }
    svg.appendChild(g)
  }

  function clearHandles(): void {
    svg.querySelector('.ws-handles-group')?.remove()
  }

  document.addEventListener('mousemove', (e: MouseEvent) => {
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
    } else {
      switch (dragState.handle) {
        case 'nw': x = sr.x + dx; y = sr.y + dy; w = sr.w - dx; h = sr.h - dy; break
        case 'n':                  y = sr.y + dy;                h = sr.h - dy; break
        case 'ne':                 y = sr.y + dy; w = sr.w + dx; h = sr.h - dy; break
        case 'e':                                 w = sr.w + dx;                break
        case 'se':                                w = sr.w + dx; h = sr.h + dy; break
        case 's':                                               h = sr.h + dy; break
        case 'sw': x = sr.x + dx;                w = sr.w - dx; h = sr.h + dy; break
        case 'w':  x = sr.x + dx;                w = sr.w - dx;                break
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

  document.addEventListener('mouseup', () => { dragState = null }, { signal: ac.signal })

  // Controls row
  const controls = document.createElement('div')
  controls.className = 'ws-viewer-controls'
  viewer.appendChild(controls)

  const detectBtn = document.createElement('button')
  detectBtn.type = 'button'
  detectBtn.className = 'ws-detect-btn'
  detectBtn.textContent = 'Detect Bubbles'
  controls.appendChild(detectBtn)

  const ocrBtn = document.createElement('button')
  ocrBtn.type = 'button'
  ocrBtn.className = 'ws-ocr-btn'
  ocrBtn.textContent = 'OCR All'
  ocrBtn.disabled = true
  controls.appendChild(ocrBtn)

  const translateBtn = document.createElement('button')
  translateBtn.type = 'button'
  translateBtn.className = 'ws-translate-btn'
  translateBtn.textContent = 'Translate All'
  translateBtn.disabled = true
  controls.appendChild(translateBtn)

  const copyPromptBtn = document.createElement('button')
  copyPromptBtn.type = 'button'
  copyPromptBtn.className = 'ws-copy-prompt-btn'
  copyPromptBtn.textContent = 'Copy Prompt'
  copyPromptBtn.disabled = true
  controls.appendChild(copyPromptBtn)

  const pasteResponseBtn = document.createElement('button')
  pasteResponseBtn.type = 'button'
  pasteResponseBtn.className = 'ws-paste-btn'
  pasteResponseBtn.textContent = 'Paste Response'
  pasteResponseBtn.disabled = true
  controls.appendChild(pasteResponseBtn)

  const inpaintBtn = document.createElement('button')
  inpaintBtn.type = 'button'
  inpaintBtn.className = 'ws-inpaint-btn'
  inpaintBtn.textContent = 'Inpaint All'
  inpaintBtn.disabled = true
  controls.appendChild(inpaintBtn)

  const typesetBtn = document.createElement('button')
  typesetBtn.type = 'button'
  typesetBtn.className = 'ws-typeset-btn'
  typesetBtn.textContent = 'Typeset All'
  typesetBtn.disabled = true
  controls.appendChild(typesetBtn)

  const previewBtn = document.createElement('button')
  previewBtn.type = 'button'
  previewBtn.className = 'ws-preview-btn'
  previewBtn.textContent = 'Preview'
  previewBtn.title = 'Hide selection overlays to preview typeset output'
  controls.appendChild(previewBtn)

  previewBtn.addEventListener('click', () => {
    const on = root.classList.toggle('ws-preview-mode')
    previewBtn.textContent = on ? 'Exit Preview' : 'Preview'
    previewBtn.classList.toggle('is-active', on)
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
  panelHeader.innerHTML = '<span class="ws-panel-title">Bubbles</span>'
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
        if (field === 'translated_zh' &&
            (bubble.state === 'detected' || bubble.state === 'ocr_done')) {
          bubble.state = 'translated'
          updateListBadge(listEl, id, bubble.state)
          updateEditorBadge(editorContainer, bubble.state)
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
    })
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

  // ── Translate button state helper ─────────────────────────────────────────

  function updateTranslateBtn(): void {
    const hasOcr = bubbles.some(b => b.raw_ja.length > 0)
    const config = loadAPIConfig()
    translateBtn.disabled = !hasOcr || config === null
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
      bubbles = await detectBubbles(
        page.imageBlob,
        (stage, _value) => { statusEl.textContent = stage },
      )
      sortedIds = sortBubbleIds(bubbles)
      selectedId = null

      countEl.textContent = String(bubbles.length)
      statusEl.textContent = `${bubbles.length} text regions found`

      rebuildSvgOverlay(bubbles, svg, (id, e) => {
        selectBubble(id)
        dragState = {
          mode: 'move', id, handle: '',
          startPct: clientToSvgPct(e),
          startRect: { ...bubbles.find(b => b.id === id)!.rect },
        }
      })
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
              if (field === 'translated_zh' &&
                  (bubble.state === 'detected' || bubble.state === 'ocr_done')) {
                bubble.state = 'translated'
                updateListBadge(listEl, id, bubble.state)
                updateEditorBadge(editorContainer, bubble.state)
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
          })
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

  translateBtn.addEventListener('click', async () => {
    const config = loadAPIConfig()
    if (!config) {
      statusEl.textContent = 'No API key — configure one in Settings'
      openSettings(() => updateTranslateBtn())
      return
    }
    translateBtn.disabled = true
    const ocrBubbles = bubbles.filter(b => b.raw_ja.length > 0)
    statusEl.textContent = 'Translating…'
    try {
      const results = await translatePage(
        ocrBubbles,
        config.providerId,
        config.key,
        stage => { statusEl.textContent = stage },
      )
      for (const { id, translated_zh } of results) {
        const bubble = bubbles.find(b => b.id === id)
        if (!bubble) continue
        bubble.translated_zh = translated_zh
        bubble.state = 'translated'
        updateListBadge(listEl, id, bubble.state)
        if (selectedId === id) selectBubble(id)
      }
      statusEl.textContent = `Translated ${results.length} bubbles`
    } catch (err) {
      statusEl.textContent = `Translation error: ${String(err).slice(0, 80)}`
      console.error(err)
    }
    updateTranslateBtn()
    updateTypesetBtn()
  })

  // ── Copy Prompt button ─────────────────────────────────────────────────────

  copyPromptBtn.addEventListener('click', async () => {
    const ocrBubbles = bubbles.filter(b => b.raw_ja.length > 0)
    try {
      await navigator.clipboard.writeText(buildPrompt(ocrBubbles))
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
      )

      // Write expanded bubble interior rects back into each speech bubble
      for (const { id, rect } of expandedRects) {
        const b = bubbles.find(b => b.id === id)
        if (b) b.bubble_rect = rect
      }

      // resultBlob is a transparent PNG overlay — speech bubble text rects are white,
      // background text regions have LaMa-reconstructed pixels; rest is alpha=0.
      // Clear first (handles re-runs), then stamp the whole blob at once.
      // The original image (img.src / page.imageBlob) is never modified.
      const inpaintBitmap = await createImageBitmap(resultBlob)
      const ctx = inpaintCanvas.getContext('2d')!
      ctx.clearRect(0, 0, inpaintCanvas.width, inpaintCanvas.height)
      ctx.drawImage(inpaintBitmap, 0, 0, inpaintCanvas.width, inpaintCanvas.height)
      inpaintBitmap.close()

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
    renderTypeset(bubbles, typesetSvg)
    statusEl.textContent = `Typeset ${bubbles.filter(b => b.translated_zh.trim()).length} bubbles`
  })
}
