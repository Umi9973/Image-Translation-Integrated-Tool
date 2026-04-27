# MangaVibe

**Browser-based manga translation pipeline — no server, no install, no cost.**
**基于浏览器的漫画翻译工具——无需服务器、无需安装、完全免费。**

---

## What is this · 这是什么

MangaVibe is a fully client-side tool that takes a raw manga page and produces a translated, typeset version ready to read. Everything runs in your browser — detection, inpainting, OCR, translation, and typesetting all happen locally with no data sent to any server.

MangaVibe 是一款完全在浏览器端运行的漫画翻译工具。上传一张漫画图源，即可得到翻译排版后的成品。OCR、翻译、嵌字全部在本地完成，不上传任何数据到服务器。

---

## Pipeline · 处理流程

```
Upload image → Detect bubbles → Inpaint (remove text) → OCR → Translate → Typeset → Download
上传图片    →   检测气泡      →   去字（LaMa修复）    → OCR →   翻译    →   排版   →  下载
```

| Stage · 阶段 | Technology · 技术 |
|---|---|
| Bubble detection · 气泡检测 | YOLOv5 + UNet (ONNX, runs in browser) |
| Text removal · 文字去除 | LaMa inpainting (ONNX, runs in browser) |
| OCR | Web Worker pipeline |
| Translation · 翻译 | Pluggable API (e.g. DeepL, GPT) |
| Typesetting · 排版 | SVG-based vertical/horizontal renderer |

---

## Key Features · 主要功能

- **Fully offline inference** — ONNX Runtime Web runs YOLOv5 and LaMa entirely in the browser via WebAssembly. No GPU server required.
- **完全离线推理** — 通过 WebAssembly 在浏览器中直接运行 YOLOv5 和 LaMa，无需 GPU 服务器。

- **Double-bubble detection** — custom gap-voting algorithm splits conjoined speech bubbles using per-row gap analysis and consensus voting.
- **双格气泡检测** — 自研逐行间隙投票算法，通过间隙分析与多数投票自动拆分连体气泡。

- **Freehand lasso tool** — draw any polygon to define a custom inpaint region for irregular bubble shapes.
- **自由套索工具** — 手绘任意多边形，为不规则气泡形状自定义修复区域。

- **Vertical CJK typesetting** — native vertical text rendering with BudouX-based line breaking and optional ruby annotation support.
- **竖排中文排版** — 原生竖排文字渲染，支持 BudouX 智能断行及注音标注。

- **Non-destructive workflow** — every bubble can be reverted and re-inpainted independently without reprocessing the full page.
- **非破坏性工作流** — 每个气泡均可独立撤销并重新修复，无需重新处理整页。

---

## Tech Stack · 技术栈

- **Runtime**: TypeScript, Vite, Web Workers
- **ML inference**: ONNX Runtime Web 1.24 (WebAssembly backend)
- **Models**: [comic-text-detector](https://huggingface.co/mayocream/comic-text-detector-onnx) (YOLOv5s + UNet), LaMa
- **Text**: BudouX (line breaking), SVG foreign object rendering
- **Storage**: Dexie (IndexedDB wrapper) for session persistence

---

## Run locally · 本地运行

```bash
npm install
npm run dev
```

Requires Node 18+. The ONNX model (~90 MB) is fetched from HuggingFace on first run and cached by the browser.

需要 Node 18+。ONNX 模型（约 90 MB）首次运行时从 HuggingFace 自动下载并缓存在浏览器中。

---

## Architecture Notes · 架构说明

All heavy computation runs in dedicated Web Workers to keep the UI thread responsive. The detect worker loads the ONNX session once and reuses it across pages. The inpaint worker receives the image blob and a bubble descriptor list, runs LaMa for background text and a white-fill pass for speech bubbles, and returns a transparent PNG overlay.

所有重计算均在独立 Web Worker 中运行，确保 UI 线程不阻塞。检测 Worker 只加载一次 ONNX 会话并复用。修复 Worker 接收图片和气泡描述列表，对背景文字运行 LaMa，对对话框执行白色填充，最终返回透明 PNG 叠加层。

The double-bubble seam detector uses two strategies: a **horizontal projection profile** (for top-bottom splits) that finds near-zero density bands in the text mask, and a **per-row gap voting** algorithm (for left-right splits) where each row independently votes for the largest dominant gap, with the median accepted only if ≥20% of rows agree.

双格气泡检测器采用两种策略：**水平投影剖面法**（用于上下分割）在文字掩码中寻找近零密度带；**逐行间隙投票法**（用于左右分割）让每一行独立投票选出最大主导间隙，仅当 ≥20% 的行达成共识时才接受该接缝。

---

## Status · 项目状态

Active development. Still Ongoing

积极开发中。
