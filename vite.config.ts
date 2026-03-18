import { defineConfig } from 'vite'
import { writeFileSync } from 'fs'
import { join } from 'path'

export default defineConfig({
  plugins: [
    {
      name: 'debug-dump',
      configureServer(server) {
        server.middlewares.use('/__debug', (req, res) => {
          if (req.method !== 'POST') { res.statusCode = 405; res.end(); return }
          const chunks: Buffer[] = []
          req.on('data', c => chunks.push(c))
          req.on('end', () => {
            try {
              const json = JSON.parse(Buffer.concat(chunks).toString())
              writeFileSync(join(process.cwd(), 'detect-debug.json'), JSON.stringify(json, null, 2))
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
