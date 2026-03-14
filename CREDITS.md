# Credits

## ML Models

| Model | Source | License | Usage |
|---|---|---|---|
| comic-text-detector | [dmMaze/comic-text-detector](https://github.com/dmMaze/comic-text-detector) | Apache-2.0 | Text block detection (YOLO + segmentation mask) |
| comic-text-detector ONNX | [mayocream/comic-text-detector-onnx](https://huggingface.co/mayocream/comic-text-detector-onnx) | Apache-2.0 | ONNX export of the above, used for in-browser inference |
| manga-ocr | [kha-white/manga-ocr](https://github.com/kha-white/manga-ocr) | Apache-2.0 | Japanese OCR (ViT encoder + BERT decoder) |
| manga-ocr-2025 ONNX | [l0wgear/manga-ocr-2025-onnx](https://huggingface.co/l0wgear/manga-ocr-2025-onnx) | Apache-2.0 | ONNX export of manga-ocr, used for in-browser inference |
| LaMa | [saic-mdal/lama](https://github.com/saic-mdal/lama) | Apache-2.0 | Large Mask inpainting model |
| LaMa ONNX | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) | Apache-2.0 | ONNX export of LaMa, used for in-browser inpainting |

## Open Source Libraries

| Library | Source | License | Usage |
|---|---|---|---|
| onnxruntime-web | [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) | MIT | WASM-based ONNX model inference in the browser |
| Dexie.js | [dexie/Dexie.js](https://github.com/dexie/Dexie.js) | Apache-2.0 | IndexedDB wrapper for local project persistence |
| Vite | [vitejs/vite](https://github.com/vitejs/vite) | MIT | Build tool and dev server |

## Reference Projects

| Project | Source | Notes |
|---|---|---|
| BallonsTranslator | [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) | Full manga translation pipeline (Python desktop app); reference for detection + inpaint + typeset workflow |
| manga-image-translator | [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator) | Original project the comic-text-detector was trained for |
| koharu | [mayocream/koharu](https://github.com/mayocream/koharu) | Rust-based manga translator using same model stack |
