import './app.css'
import { renderWorkspace } from './workspace'
import type { MangaPage } from '../types'

export function renderApp(container: HTMLElement): void {
  container.innerHTML = `
    <div class="app">
      <header class="topbar">
        <span class="logo">Kalar</span>
        <span class="tagline">Scanlation Studio</span>
      </header>
      <main class="workspace">
        <div class="upload-zone" id="upload-zone">
          <p>Drop a manga page here or <label for="file-input">browse</label></p>
          <input id="file-input" type="file" accept="image/*" hidden />
        </div>
      </main>
    </div>
  `

  const zone = container.querySelector<HTMLDivElement>('#upload-zone')!
  const fileInput = container.querySelector<HTMLInputElement>('#file-input')!

  zone.querySelector('label')!.style.cursor = 'pointer'
  zone.addEventListener('click', () => fileInput.click())

  zone.addEventListener('dragover', (e) => {
    e.preventDefault()
    zone.classList.add('drag-over')
  })
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'))
  zone.addEventListener('drop', (e) => {
    e.preventDefault()
    zone.classList.remove('drag-over')
    const file = e.dataTransfer?.files[0]
    if (file) handleFile(container, file)
  })

  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0]
    if (file) handleFile(container, file)
  })
}

function handleFile(container: HTMLElement, file: File): void {
  const page: MangaPage = {
    id: crypto.randomUUID(),
    projectId: 'session',
    filename: file.name,
    imageBlob: file,
    bubbles: [],
    createdAt: new Date(),
  }
  renderWorkspace(container, page)
}
