import { defineConfig } from 'vite'

export default defineConfig({
  // Allow SharedArrayBuffer for Transformers.js Web Workers
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
})
