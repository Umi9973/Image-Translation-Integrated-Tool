/**
 * dict-panel.ts — User dictionary panel (left sidebar)
 *
 * Two dictionaries:
 *   - No-Split (seg dict): session-only, injected directly into a cloned BudouX
 *     model via addPhraseToParser(). Resets on page refresh automatically;
 *     also has a manual Reset button. NOT stored in localStorage.
 *   - Translation glossary: persisted in localStorage. JP→ZH overrides injected
 *     into the LLM prompt at translate/copy-prompt time.
 */

import { addPhraseToParser, resetParser } from '../pipeline/typeset'

const GLOSSARY_KEY = 'mangavibe_glossary'

export interface GlossaryEntry { ja: string; zh: string }

// ── Persistence (glossary only) ───────────────────────────────────────────────

function loadGlossary(): GlossaryEntry[] {
  try { return JSON.parse(localStorage.getItem(GLOSSARY_KEY) ?? '[]') as GlossaryEntry[] } catch { return [] }
}
function saveGlossary(g: GlossaryEntry[]): void { localStorage.setItem(GLOSSARY_KEY, JSON.stringify(g)) }

// ── Module-level state ────────────────────────────────────────────────────────
// segDict is intentionally NOT persisted — session-only, resets on refresh

let segDict: string[]          = []
let glossary: GlossaryEntry[]  = loadGlossary()

export function getGlossary(): GlossaryEntry[] { return glossary }

// ── Render ───────────────────────────────────────────────────────────────────

type DictTab = 'seg' | 'glossary'

export function renderDictPanel(container: HTMLElement): void {
  let tab: DictTab = 'seg'

  container.innerHTML = ''

  // Header: title + tab selector
  const header = document.createElement('div')
  header.className = 'ws-dict-header'

  const title = document.createElement('span')
  title.className = 'ws-dict-title'
  title.textContent = 'Dictionary'

  const tabSelect = document.createElement('select')
  tabSelect.className = 'ws-shape-select'
  tabSelect.innerHTML = '<option value="seg">No-Split</option><option value="glossary">Glossary</option>'
  tabSelect.addEventListener('change', () => {
    tab = tabSelect.value as DictTab
    rebuildAddRow()
    rebuildList()
  })

  header.appendChild(title)
  header.appendChild(tabSelect)
  container.appendChild(header)

  // Add-entry row
  const addRow = document.createElement('div')
  addRow.className = 'ws-dict-add-row'
  container.appendChild(addRow)

  // Entry list
  const listEl = document.createElement('div')
  listEl.className = 'ws-dict-list'
  container.appendChild(listEl)

  function rebuildAddRow(): void {
    addRow.innerHTML = ''
    if (tab === 'seg') {
      const input = document.createElement('input')
      input.type = 'text'
      input.className = 'ws-dict-input'
      input.placeholder = 'Keep phrase whole…'

      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'ws-dict-add-btn'
      btn.textContent = '+'

      btn.addEventListener('click', () => {
        const phrase = input.value.trim()
        if (!phrase || segDict.includes(phrase)) return
        segDict = [...segDict, phrase]
        addPhraseToParser(phrase)   // inject into BudouX model clone
        input.value = ''
        rebuildList()
      })
      input.addEventListener('keydown', e => { if (e.key === 'Enter') btn.click() })

      addRow.appendChild(input)
      addRow.appendChild(btn)
    } else {
      const jaInput = document.createElement('input')
      jaInput.type = 'text'
      jaInput.className = 'ws-dict-input ws-dict-input--half'
      jaInput.placeholder = 'JP term'

      const zhInput = document.createElement('input')
      zhInput.type = 'text'
      zhInput.className = 'ws-dict-input ws-dict-input--half'
      zhInput.placeholder = 'ZH override'

      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'ws-dict-add-btn'
      btn.textContent = '+'

      btn.addEventListener('click', () => {
        const ja = jaInput.value.trim()
        const zh = zhInput.value.trim()
        if (!ja || !zh) return
        if (glossary.some(e => e.ja === ja)) {
          glossary = glossary.map(e => e.ja === ja ? { ja, zh } : e)
        } else {
          glossary = [...glossary, { ja, zh }]
        }
        saveGlossary(glossary)
        jaInput.value = ''
        zhInput.value = ''
        rebuildList()
      })
      zhInput.addEventListener('keydown', e => { if (e.key === 'Enter') btn.click() })

      addRow.appendChild(jaInput)
      addRow.appendChild(zhInput)
      addRow.appendChild(btn)
    }
  }

  function rebuildList(): void {
    listEl.innerHTML = ''
    if (tab === 'seg') {
      if (segDict.length === 0) {
        listEl.appendChild(makeEmpty('No entries — add phrases\nBudouX should never split'))
        return
      }
      for (const phrase of segDict) {
        listEl.appendChild(makeItem(phrase, () => {
          // Removing a single entry requires rebuilding the whole model clone
          // since BW2 scores can't be "un-added" individually
          segDict = segDict.filter(p => p !== phrase)
          resetParser()
          segDict.forEach(p => addPhraseToParser(p))
          rebuildList()
        }))
      }
      // Reset button — clears all phrases and restores default BudouX model
      const resetBtn = document.createElement('button')
      resetBtn.type = 'button'
      resetBtn.className = 'ws-dict-reset-btn'
      resetBtn.textContent = 'Reset all'
      resetBtn.addEventListener('click', () => {
        segDict = []
        resetParser()
        rebuildList()
      })
      listEl.appendChild(resetBtn)
    } else {
      if (glossary.length === 0) {
        listEl.appendChild(makeEmpty('No entries — add JP→ZH\nterm overrides for the LLM'))
        return
      }
      for (const entry of glossary) {
        listEl.appendChild(makeItem(`${entry.ja} → ${entry.zh}`, () => {
          glossary = glossary.filter(e => e.ja !== entry.ja)
          saveGlossary(glossary)
          rebuildList()
        }))
      }
    }
  }

  rebuildAddRow()
  rebuildList()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function makeEmpty(msg: string): HTMLDivElement {
  const el = document.createElement('div')
  el.className = 'ws-dict-empty'
  el.textContent = msg
  return el
}

function makeItem(label: string, onDelete: () => void): HTMLDivElement {
  const item = document.createElement('div')
  item.className = 'ws-dict-item'

  const text = document.createElement('span')
  text.className = 'ws-dict-item-text'
  text.textContent = label
  text.title = label

  const del = document.createElement('button')
  del.type = 'button'
  del.className = 'ws-delete-btn'
  del.style.opacity = '1'
  del.textContent = '×'
  del.addEventListener('click', onDelete)

  item.appendChild(text)
  item.appendChild(del)
  return item
}
