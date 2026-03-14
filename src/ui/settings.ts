import './settings.css'
import {
  TRANSLATION_PROVIDERS,
  loadAPIConfig,
  saveAPIConfig,
  clearAPIConfig,
} from '../pipeline/translate'

export function openSettings(onConfigChange?: () => void): void {
  // Remove any existing modal first
  document.getElementById('st-overlay')?.remove()

  const overlay = document.createElement('div')
  overlay.id = 'st-overlay'
  overlay.className = 'st-overlay'

  const modal = document.createElement('div')
  modal.className = 'st-modal'
  overlay.appendChild(modal)

  // ── Header ────────────────────────────────────────────────────────────────

  const header = document.createElement('div')
  header.className = 'st-header'

  const title = document.createElement('span')
  title.className = 'st-title'
  title.textContent = 'Settings'
  header.appendChild(title)

  const closeBtn = document.createElement('button')
  closeBtn.type = 'button'
  closeBtn.className = 'st-close-btn'
  closeBtn.textContent = '✕'
  closeBtn.addEventListener('click', () => overlay.remove())
  header.appendChild(closeBtn)

  modal.appendChild(header)

  // ── Body ──────────────────────────────────────────────────────────────────

  const body = document.createElement('div')
  body.className = 'st-body'
  modal.appendChild(body)

  // ── Section 1: BYOK API ──────────────────────────────────────────────────

  const section1 = document.createElement('div')
  section1.className = 'st-section'
  body.appendChild(section1)

  const s1Title = document.createElement('div')
  s1Title.className = 'st-section-title'
  s1Title.textContent = 'Developer Mode — Bring Your Own API Key'
  section1.appendChild(s1Title)

  // Provider selector row (label + "Get API key →" link)
  const providerRow = document.createElement('div')
  providerRow.className = 'st-label-row'
  const providerLabel = document.createElement('span')
  providerLabel.className = 'st-label'
  providerLabel.textContent = 'Provider'
  const apiKeyLink = document.createElement('a')
  apiKeyLink.className = 'st-get-key-link'
  apiKeyLink.target = '_blank'
  apiKeyLink.rel = 'noopener noreferrer'
  apiKeyLink.textContent = 'Get API key →'
  providerRow.appendChild(providerLabel)
  providerRow.appendChild(apiKeyLink)
  section1.appendChild(providerRow)

  const providerSelect = document.createElement('select')
  providerSelect.className = 'st-select'
  for (const p of TRANSLATION_PROVIDERS) {
    const opt = document.createElement('option')
    opt.value = p.id
    opt.textContent = p.name
    providerSelect.appendChild(opt)
  }
  section1.appendChild(providerSelect)

  // API key input row (label + saved indicator)
  const keyRow = document.createElement('div')
  keyRow.className = 'st-label-row'
  const keyLabel = document.createElement('span')
  keyLabel.className = 'st-label'
  keyLabel.textContent = 'API Key'
  const savedIndicator = document.createElement('span')
  savedIndicator.className = 'st-saved-indicator'
  keyRow.appendChild(keyLabel)
  keyRow.appendChild(savedIndicator)
  section1.appendChild(keyRow)

  const keyInput = document.createElement('input')
  keyInput.type = 'password'
  keyInput.className = 'st-input'
  keyInput.placeholder = 'sk-...'
  keyInput.autocomplete = 'off'
  section1.appendChild(keyInput)

  // Helper: sync the "Get API key" link and saved indicator to current state
  function syncProviderUI(): void {
    const p = TRANSLATION_PROVIDERS.find(pr => pr.id === providerSelect.value)
    if (p) apiKeyLink.href = p.apiKeyUrl
    const cfg = loadAPIConfig()
    if (cfg && cfg.providerId === providerSelect.value) {
      savedIndicator.textContent = '✓ key saved'
      savedIndicator.className = 'st-saved-indicator st-saved-indicator--ok'
    } else {
      savedIndicator.textContent = ''
      savedIndicator.className = 'st-saved-indicator'
    }
  }

  providerSelect.addEventListener('change', syncProviderUI)

  // Restore saved config
  const existing = loadAPIConfig()
  if (existing) {
    if (TRANSLATION_PROVIDERS.find(p => p.id === existing.providerId)) {
      providerSelect.value = existing.providerId
    }
    keyInput.value = existing.key
  }
  syncProviderUI()

  // Save / Clear row
  const btnRow = document.createElement('div')
  btnRow.className = 'st-btn-row'
  section1.appendChild(btnRow)

  const statusMsg = document.createElement('span')
  statusMsg.className = 'st-status'

  const saveBtn = document.createElement('button')
  saveBtn.type = 'button'
  saveBtn.className = 'st-save-btn'
  saveBtn.textContent = 'Save Key'
  saveBtn.addEventListener('click', () => {
    const key = keyInput.value.trim()
    if (!key) { statusMsg.textContent = 'Please enter an API key.'; return }
    saveAPIConfig(providerSelect.value, key)
    statusMsg.textContent = 'Key saved.'
    syncProviderUI()
    onConfigChange?.()
  })
  btnRow.appendChild(saveBtn)

  const clearBtn = document.createElement('button')
  clearBtn.type = 'button'
  clearBtn.className = 'st-clear-btn'
  clearBtn.textContent = 'Clear Key'
  clearBtn.addEventListener('click', () => {
    clearAPIConfig()
    keyInput.value = ''
    statusMsg.textContent = 'Key cleared.'
    syncProviderUI()
    onConfigChange?.()
  })
  btnRow.appendChild(clearBtn)
  btnRow.appendChild(statusMsg)

  // Privacy note
  const privacy = document.createElement('div')
  privacy.className = 'st-privacy'
  privacy.innerHTML = `
    <strong>How we handle your key:</strong>
    <ul>
      <li>Stored only in this browser's <code>localStorage</code> — never sent to MangaVibe servers.</li>
      <li>Verify storage: DevTools → Application → Local Storage → look for <code>mangavibe_api_config</code>.</li>
      <li>Verify the API call: DevTools → Network tab → when you click "Translate All", you should see a request going directly to <code>api.openai.com</code> or <code>api.deepseek.com</code> with no MangaVibe domain in the chain.</li>
      <li>MangaVibe is a static site with no backend — there is no server to receive your key.</li>
    </ul>
  `
  section1.appendChild(privacy)

  // ── Section 2: Link-out ──────────────────────────────────────────────────

  const divider = document.createElement('hr')
  divider.className = 'st-divider'
  body.appendChild(divider)

  const section2 = document.createElement('div')
  section2.className = 'st-section'
  body.appendChild(section2)

  const s2Title = document.createElement('div')
  s2Title.className = 'st-section-title'
  s2Title.textContent = 'No-Account Mode — Use ChatGPT or DeepSeek Web'
  section2.appendChild(s2Title)

  const steps = document.createElement('ol')
  steps.className = 'st-steps'
  steps.innerHTML = `
    <li>Run <strong>OCR All</strong> so every bubble has Japanese text.</li>
    <li>Click <strong>Copy Prompt</strong> in the toolbar — copies the full translation prompt.</li>
    <li>Open your AI chat below, paste the prompt, and wait for the response.</li>
    <li>Copy the <em>entire</em> response (the JSON array must be included).</li>
    <li>Back in MangaVibe, click <strong>Paste Response</strong> — translations are applied automatically.</li>
  `
  section2.appendChild(steps)

  const linkRow = document.createElement('div')
  linkRow.className = 'st-link-row'
  for (const p of TRANSLATION_PROVIDERS) {
    const link = document.createElement('a')
    link.href = p.chatUrl
    link.target = '_blank'
    link.rel = 'noopener noreferrer'
    link.className = 'st-chat-link'
    link.textContent = `Open ${p.name.split(' ')[0]}`
    linkRow.appendChild(link)
  }
  section2.appendChild(linkRow)

  // Close on backdrop click
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.remove()
  })

  document.body.appendChild(overlay)
}
