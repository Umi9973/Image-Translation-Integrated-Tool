# MangaVibe: Local-First Scanlation & Review Studio

## 🎯 Project Vision
A professional-grade, browser-based workstation for Japanese-to-Chinese manga translation.
Unlike "Readers" (e.g., Mokuro), this is an **Editor** designed for scanlation workflows: Detect -> OCR -> Translate -> Review -> Inpaint -> Typeset.

## 🛠️ Multi-Stage Pipeline (Decoupled)
| # | Stage | Model / Tool | Status |
|---|---|---|---|
| 1 | **Detection** | `mayocream/comic-text-detector-onnx` (YOLO) via onnxruntime-web | ✅ Live |
| 2 | **OCR** | `l0wgear/manga-ocr-2025-onnx` (ViT + BERT decoder) via onnxruntime-web | ✅ Live |
| 3 | **Translation** | BYOK API (OpenAI/DeepSeek direct browser call) **or** link-out copy/paste mode | ✅ Live |
| 4 | **Inpainting** | Three-way routing per bubble: bright interior → white fill (speech bubble); dark + uniform border → solid color fill (toned panel text); dark + complex border → manga-tuned LaMa ONNX (`dreMaz/AnimeMangaInpainting/lama_manga_fp32.onnx`, ~199 MB, served from `public/`, OPFS-cached, auto-recovers corrupt cache) reconstructs background; uses detection heatmap for pixel-accurate text masks; transparent PNG overlay output | ✅ Live |
| 5 | **Typesetting** | SVG overlay — vertical CJK columns (`writing-mode="vertical-rl"`), right-to-left, auto-fit; uses `bubble_rect` (full bubble interior) when available; `\` in translation forces column-group break; font: ZCOOL KuaiLe | ✅ Live |

## 🏗️ Architecture: Zero-Server / Privacy-First
- **No Backend:** Static frontend export. No user images or keys ever touch our servers.
- **Local Storage:** Use **IndexedDB (via Dexie.js)** for project management and **OPFS** for high-performance image caching.
- **BYOK (Bring Your Own Key):** API key stored in browser `localStorage` (`mangavibe_api_config`). Supports OpenAI and DeepSeek (OpenAI-compatible endpoints). Anthropic excluded — no browser CORS support. Link-out mode available for users without an API account.

## 📜 Coding Rules & "The Vibe"
- **State Sovereignty:** Every bubble is a unique object with states: `detected`, `ocr_done`, `translated`, `reviewed`.
- **Human-in-the-Loop:** UI must allow editing at *every* stage (e.g., fix the OCR before paying for the Translation).
- **Performance:** Offload OCR and Inpainting to **Web Workers** to prevent UI freezing.
- **Coordinate Integrity:** Maintain original image aspect ratios. Use a "Glass Table" (SVG) overlay for interactive bubble selection.
- **Reading Order:** Sort detected bubbles Right-to-Left, Top-to-Bottom.

## 📁 Project File Map
```
project-2/
├── index.html                  ← app entry point (root, NOT in src/)
├── vite.config.ts              ← Vite config; sets COOP/COEP headers for SharedArrayBuffer
├── tsconfig.json               ← TypeScript config; target ES2022
├── package.json                ← scripts: dev / build / preview
├── CLAUDE.md                   ← this file
├── ISSUES.md                   ← bug post-mortems (add when asked)
├── src/
│   ├── main.ts                 ← mounts renderApp() into #app
│   ├── types/
│   │   └── index.ts            ← ALL shared types (MangaBubble, MangaPage, BubbleState)
│   ├── db/
│   │   └── index.ts            ← Dexie.js instance; schema: projects + pages tables
│   ├── pipeline/
│   │   ├── detect.ts           ← ✅ detectBubbles(blob, onProgress?) → MangaBubble[]
│   │   ├── ocr.ts              ← ✅ runOCR(bubble, imageBlob, onProgress?) → string
│   │   ├── translate.ts        ← ✅ BYOK API + link-out; provider registry, buildPrompt, translatePage, parseTranslationResponse, localStorage config
│   │   ├── inpaint.ts          ← ✅ inpaintPage(bubbles, imageBlob, onProgress?) → Blob
│   │   └── typeset.ts          ← ✅ renderTypeset(bubbles, svg) → void; auto-fits CJK text into each bubble rect
│   ├── workers/
│   │   ├── detect.worker.ts    ← ✅ comic-text-detector ONNX inference; outputs tight text-pixel rects (tightenToMask); double-bubble seam detection (findSeamY / findSeamX) with false-positive rejection (gap + perpendicular-axis overlap check) + three-pass deduplication (wrapper / containment / coverage)
│   │   ├── ocr.worker.ts       ← ✅ manga-ocr encoder+decoder ONNX inference
│   │   └── inpaint.worker.ts   ← ✅ Three-way inpaint routing: isBrightRegion() (interior 5×5) → white; sampleBorderColor() stddev check → solid fill; else LaMa ONNX; OPFS cache with auto-recovery on corruption
│   └── ui/
│       ├── app.ts              ← upload screen; on file pick → renderWorkspace()
│       ├── app.css             ← global dark theme; CSS variables in :root
│       ├── workspace.ts        ← ✅ workspace controller; owns bubbles/sortedIds/selectedId; wires all pipeline buttons
│       ├── workspace.css       ← workspace layout + component styles
│       ├── settings.ts         ← ✅ openSettings() modal; BYOK key management + link-out workflow
│       └── settings.css        ← settings modal overlay styles
└── .venv/                      ← Python 3.13 env (tooling only, not part of the app)
```

## 🧰 Tech Stack
| Concern | Tool |
|---|---|
| Bundler / Dev server | Vite 6 |
| Language | TypeScript 5 (strict mode) |
| DB / Persistence | Dexie.js 4 (IndexedDB wrapper) |
| Image cache | OPFS (Origin Private File System) |
| ML inference | onnxruntime-web 1.24.2 (WASM backend, models fetched from HuggingFace) |
| Inpainting | LaMa ONNX via onnxruntime-web (`Carve/LaMa-ONNX/lama_fp32.onnx`, ~208 MB, OPFS-cached after first use) |
| Hosting target | Static (Vercel / Netlify / GitHub Pages) |

## 🚫 Git Rules
- **NEVER run `git push` without explicit user permission.** Always stop and ask before pushing to GitHub.

## 🚦 Dev Commands
```bash
npm run dev       # start local dev server at localhost:5173
npm run build     # type-check + production bundle → dist/
npm run preview   # preview the production build locally
```

## 🗂️ Bubble Data Structure
```typescript
interface MangaBubble {
  id: string;
  rect: { x: number; y: number; w: number; h: number };        // Percentage-based, tight text bbox (from detection)
  bubble_rect?: { x: number; y: number; w: number; h: number }; // Percentage-based, full bubble interior (set after inpaint)
  raw_ja: string;       // Result of OCR
  translated_zh: string; // Result of LLM; '\' = forced column-group break in typesetting
  is_locked: boolean;   // User has finalized this bubble
  layer_z: number;      // For overlapping bubbles
}
```
