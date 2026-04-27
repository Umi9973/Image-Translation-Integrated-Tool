import './settings.css'
import { TRANSLATION_PROVIDERS } from '../pipeline/translate'

export function openSettings(): void {
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
    <li>Back in Kalar, click <strong>Paste Response</strong> — translations are applied automatically.</li>
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
