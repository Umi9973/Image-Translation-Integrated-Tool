import type { MangaBubble } from '../types'

export interface GlossaryEntry { ja: string; zh: string }

export interface ProviderDef {
  id: string
  name: string
  model: string
  endpoint: string   // OpenAI-compatible chat completions URL
  chatUrl: string    // Link-out URL for this provider's web UI
  apiKeyUrl: string  // Where to obtain an API key
}

// Extensible registry — add new providers here without touching other files
export const TRANSLATION_PROVIDERS: ProviderDef[] = [
  {
    id: 'openai',
    name: 'OpenAI (GPT-4o)',
    model: 'gpt-4o',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    chatUrl: 'https://chatgpt.com/',
    apiKeyUrl: 'https://platform.openai.com/api-keys',
  },
  {
    id: 'deepseek',
    name: 'DeepSeek (DeepSeek-V3)',
    model: 'deepseek-chat',
    endpoint: 'https://api.deepseek.com/chat/completions',
    chatUrl: 'https://chat.deepseek.com/',
    apiKeyUrl: 'https://platform.deepseek.com/api_keys',
  },
]

// ── Prompt building ────────────────────────────────────────────────────────

export function buildPrompt(bubbles: MangaBubble[], glossary?: GlossaryEntry[]): string {
  const bubblesJson = JSON.stringify(
    bubbles.map(b => ({ id: b.id, text: b.raw_ja })),
    null,
    2,
  )

  const glossarySection = glossary && glossary.length > 0
    ? `\n## Glossary (translate these terms exactly — do not deviate)\n${glossary.map(e => `${e.ja} → ${e.zh}`).join('\n')}\n`
    : ''

  return `You are a professional manga scanlator specialising in Japanese-to-Chinese translation.

## Phase 1 — Contextual Analysis (internal reasoning only — do NOT output this section)
Read all bubble texts together. Identify:
- Genre and setting (shounen, slice-of-life, etc.)
- Character archetypes and their speech registers
- Recurring names, terms, and onomatopoeia that need consistent handling
- Overall emotional register (dramatic, comedic, tense, etc.)

## Phase 2 — Style Commitment (output this section as a brief comment before the JSON)
Write 2–3 sentences committing to your translation choices:
- Chinese dialect / register (e.g., "使用正式繁體" vs "使用日常简体")
- How you will handle onomatopoeia and sound effects
- Any character-specific speech quirks

## Phase 3 — Translate
Translate every bubble. Rules:
- Preserve the emotional nuance and speech register of each character
- Keep translations concise — manga bubbles have limited space
- Onomatopoeia: use Chinese equivalents where natural, or romanise if no good equivalent exists
- Do NOT add explanations, footnotes, or translator notes inside translated_zh
- If a translation has a natural phrase break (e.g. between clauses or sentences), insert the two characters \\ (escaped backslash — this is required for valid JSON) at the break point. Use at most one or two breaks per bubble. Do not add a break if the text flows naturally as one unit. Example: "我来了\\\\让我们走" (the rendered break character is \\).

Output a valid JSON array wrapped in a \`\`\`json code block immediately after your style commitment comment, and nothing else after it:
\`\`\`json
[{ "id": "<bubble_id>", "translated_zh": "<chinese translation>" }, ...]
\`\`\`
${glossarySection}
## Bubbles
${bubblesJson}`
}

// ── Response parsing ───────────────────────────────────────────────────────

/** Fix unescaped backslashes in a JSON string (model sometimes outputs raw \ instead of \\). */
function fixBackslashes(json: string): string {
  // Replace \ not followed by a valid JSON escape char with \\
  return json.replace(/\\(?!["\\/bfnrtu])/g, '\\\\')
}

export function parseTranslationResponse(
  text: string,
): { id: string; translated_zh: string }[] {
  // Strip ```json ... ``` fences if present
  const fenced = text.match(/```json\s*([\s\S]*?)```/)
  if (fenced) text = fenced[1].trim()

  // Try from last '[' — model may prefix style notes before the JSON array
  const lastBracket = text.lastIndexOf('[')
  if (lastBracket !== -1) {
    const slice = text.slice(lastBracket)
    try {
      return JSON.parse(slice) as { id: string; translated_zh: string }[]
    } catch {
      try {
        return JSON.parse(fixBackslashes(slice)) as { id: string; translated_zh: string }[]
      } catch { /* fall through to greedy search */ }
    }
  }
  const match = text.match(/\[[\s\S]*\]/)
  if (!match) throw new Error('No JSON array found in response')
  try {
    return JSON.parse(match[0]) as { id: string; translated_zh: string }[]
  } catch {
    return JSON.parse(fixBackslashes(match[0])) as { id: string; translated_zh: string }[]
  }
}

// ── API translation ────────────────────────────────────────────────────────

export async function translatePage(
  bubbles: MangaBubble[],
  providerId: string,
  apiKey: string,
  onProgress?: (stage: string) => void,
  glossary?: GlossaryEntry[],
): Promise<{ id: string; translated_zh: string }[]> {
  const provider = TRANSLATION_PROVIDERS.find(p => p.id === providerId)
  if (!provider) throw new Error(`Unknown provider: ${providerId}`)

  onProgress?.('Sending request…')

  const response = await fetch(provider.endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: provider.model,
      messages: [{ role: 'user', content: buildPrompt(bubbles, glossary) }],
      temperature: 0.3,
    }),
  })

  if (!response.ok) {
    const errText = await response.text().catch(() => response.statusText)
    throw new Error(`API error ${response.status}: ${errText.slice(0, 200)}`)
  }

  onProgress?.('Parsing response…')

  const data = await response.json() as { choices?: { message?: { content?: string } }[] }
  const content = data.choices?.[0]?.message?.content ?? ''
  return parseTranslationResponse(content)
}
