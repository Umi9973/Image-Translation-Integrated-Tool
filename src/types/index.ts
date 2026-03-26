// Core bubble state machine
export type BubbleState = 'detected' | 'ocr_done' | 'translated' | 'reviewed'

export interface MangaBubble {
  id: string
  rect: { x: number; y: number; w: number; h: number }        // percentage-based, tight text bbox
  bubble_rect?: { x: number; y: number; w: number; h: number } // percentage-based, full bubble interior (set after inpaint)
  raw_ja: string        // OCR result
  translated_zh: string // LLM translation result
  state: BubbleState
  is_locked: boolean    // user has finalized this bubble
  layer_z: number       // for overlapping bubbles
  source?: 'detected' | 'manual'  // how the bubble was created
  shape?: 'rect' | 'bubble'       // cover background shape: sharp rect or heavily-rounded
  cover?: boolean                 // render background fill behind text in typeset layer
  coverOutline?: boolean          // draw a black border around the cover fill
  font_size_override?: number     // force exact font size in typeset (skips auto-fit); undefined = auto
  text_direction?: 'vertical' | 'horizontal'  // default 'vertical'; 'horizontal' renders LTR text
  text_offset_x?: number  // horizontal text position shift (% of image width, + = right)
  text_offset_y?: number  // vertical text position shift (% of image height, + = down)
  inpaint_color?: string  // hex color override for inpaint fill (e.g. '#f0e8d0'); undefined = auto-detect
  is_background?: boolean // inpaint route override: true = force solid fill, false = force bubble fill, undefined = auto
  rotation?: number       // degrees clockwise; applied to both inpaint fill and typeset text
  det_conf?: number         // detection confidence score (debug only)
  det_mask_density?: number // fraction of box pixels flagged as text by heatmap (debug only)
}

export interface MangaPage {
  id: string
  projectId: string
  filename: string
  imageBlob: Blob
  bubbles: MangaBubble[]
  createdAt: Date
}

export interface Project {
  id: string
  name: string
  createdAt: Date
  pageCount: number
}

// BYOK provider config
export type LLMProvider = 'anthropic' | 'openai' | 'deepseek'

export interface APIKeyConfig {
  provider: LLMProvider
  key: string
}
