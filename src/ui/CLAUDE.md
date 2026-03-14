# UI Module

Vanilla TypeScript + CSS. No UI framework (no React/Vue). Keep it simple.

## Files
| File | Role |
|---|---|
| `app.ts` | Upload screen; on file pick, builds `MangaPage` and calls `renderWorkspace()` |
| `app.css` | Global styles; all CSS variables defined in `:root` here |
| `workspace.ts` | Workspace controller; exports `renderWorkspace(container, page)`; owns all mutable state (bubbles, sortedIds, selectedId); wires Detect / OCR All / Translate All / Copy Prompt / Paste Response / Inpaint All / Typeset All buttons; bubble delete (× button per list item); drag-to-move + 8-handle resize on selected bubble |
| `workspace.css` | Workspace layout + component styles; uses CSS vars from `app.css` `:root` |
| `settings.ts` | Settings modal; exports `openSettings(onConfigChange?)`; manages BYOK provider/key (saved to localStorage) and link-out workflow explanation |
| `settings.css` | Settings modal overlay styles |

## CSS Variables (defined in app.css :root)
| Variable | Value | Use |
|---|---|---|
| `--bg` | `#0f0f13` | Page background |
| `--surface` | `#1a1a24` | Cards, panels, topbar |
| `--border` | `#2e2e3e` | Borders, dividers |
| `--accent` | `#7c5cfc` | Buttons, highlights, links |
| `--text` | `#e8e8f0` | Primary text |
| `--text-dim` | `#7a7a9a` | Muted / secondary text |

## Rules
- All new UI components are functions: `function renderX(container: HTMLElement): void`
- No inline styles — use CSS classes and the variables above.
- The image frame uses four stacked absolute layers (bottom→top): `<img>` (original, never modified) → `.ws-inpaint-layer` `<canvas>` (LaMa output) → `.ws-typeset-layer` SVG (Chinese text) → `.ws-bubble-overlay` SVG (selection rects).
- The inpaint canvas intrinsic size is set to `img.naturalWidth × img.naturalHeight` on image load; CSS stretches it to 100%×100% of the frame.
- The "Glass Table" SVG overlay (bubble selection layer) sits on top of the manga image; it must use `pointer-events: none` by default and `pointer-events: all` only on bubble rects.
- Image display must preserve original aspect ratio at all times (`object-fit: contain`).
- Bubble list is always sorted Right-to-Left, Top-to-Bottom before rendering.

## Drag / Resize (workspace.ts)
- **Move**: `mousedown` on a bubble rect sets `dragState.mode = 'move'`; `document.mousemove` translates `bubble.rect` clamped to `[0, 100 - w/h]`; `mouseup` clears state.
- **Resize**: when a bubble is selected, 8 `<text>` handles (Unicode arrows ↖↑↗→↘↓↙←) are rendered at the corners and edge midpoints of the bubble rect, centered on the boundary (half inside / half outside). `mousedown` on a handle sets `dragState.mode = 'resize'` with the handle id; `mousemove` adjusts the appropriate edges; minimum size 2%.
- **Coordinates**: SVG `viewBox="0 0 100 100"` maps directly to `bubble.rect` percentages. Mouse position converted via `clientToSvgPct()` using `svg.getBoundingClientRect()`.
- **Cleanup**: `AbortController` (`ac`) removes `document` listeners when "← New Page" is clicked.
- **CSS**: `.ws-handle-arrow` — orange fill+stroke, bold, `pointer-events: all`. `.ws-bubble-rect.is-selected` — `cursor: move`.
