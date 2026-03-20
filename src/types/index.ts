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
