import { defineConfig } from 'vite'
import { writeFileSync } from 'fs'
import { join } from 'path'

export default defineConfig({
  plugins: [
    {
      name: 'debug-dump',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (!req.url?.startsWith('/__debug/') || req.method !== 'POST') return next()
          const name = req.url.slice('/__debug/'.length).replace(/[^a-z0-9_-]/gi, '_')
          const chunks: Buffer[] = []
          req.on('data', c => chunks.push(c))
          req.on('end', () => {
            try {
              const json = JSON.parse(Buffer.concat(chunks).toString())
              writeFileSync(join(process.cwd(), `${name}-debug.json`), JSON.stringify(json, null, 2))
              res.statusCode = 204; res.end()
            } catch { res.statusCode = 400; res.end() }
          })
        })
      },
    },
  ],
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
