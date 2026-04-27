export type Locale = 'en' | 'zh'

const STRINGS = {
  // Topbar
  settingsBtn:        { en: '⚙ Settings',        zh: '⚙ 设置' },
  newPageBtn:         { en: '← New Page',         zh: '← 新页面' },

  // Controls
  detectBubbles:      { en: 'Detect Bubbles',     zh: '检测气泡' },
  ocrAll:             { en: 'OCR All',             zh: '识图全部' },
  copyPrompt:         { en: 'Copy Prompt',         zh: '复制提示词' },
  pasteResponse:      { en: 'Paste Response',      zh: '粘贴回复' },
  inpaintAll:         { en: 'Inpaint All',         zh: '去字' },
  revertAllInpaint:   { en: 'Revert All Inpaint',  zh: '还原去字' },
  revertAllTypeset:   { en: 'Revert All Typeset',  zh: '还原嵌字' },
  typesetAll:         { en: 'Typeset All',         zh: '嵌字' },
  preview:            { en: 'Preview',             zh: '预览' },
  exitPreview:        { en: 'Exit Preview',        zh: '退出预览' },
  download:           { en: '↓ Download',          zh: '↓ 下载' },

  // Panel
  bubblesTitle:       { en: 'Bubbles',             zh: '气泡列表' },

  // Editor
  selectBubbleHint:   { en: 'Select a bubble to edit', zh: '选择气泡以编辑' },
  jaLabel:            { en: 'Japanese (raw_ja)',   zh: '日文（原文）' },
  zhLabel:            { en: 'Translation (translated_zh)', zh: '译文' },
  prevBtn:            { en: '← Prev',             zh: '← 上一个' },
  nextBtn:            { en: 'Next →',             zh: '下一个 →' },
  lockBtn:            { en: 'Lock',               zh: '锁定' },
  unlockBtn:          { en: 'Unlock',             zh: '解锁' },
  revertInpaint:      { en: 'Revert Inpaint',     zh: '还原当前去字' },
  revertTypeset:      { en: 'Revert Typeset',     zh: '还原当前嵌字' },
  fontSizeLabel:      { en: 'Font size',          zh: '字号' },
  rotationLabel:      { en: 'Rotation°',          zh: '旋转°' },
  textColorLabel:     { en: 'Text color',         zh: '文字颜色' },
  blackBtn:           { en: '● Black',            zh: '● 黑色' },
  whiteBtn:           { en: '○ White',            zh: '○ 白色' },
  resetPosition:      { en: 'Reset position',     zh: '重置位置' },
  dragHint:           { en: '↖ Drag text on image to reposition', zh: '↖ 拖拽图上文字可移动' },

  // Cover section labels
  coverBgLabel:       { en: ' Cover background',  zh: ' 遮盖背景' },
  outlineLabel:       { en: ' Outline',           zh: ' 描边' },
  horizontalText:     { en: ' Horizontal text',   zh: ' 横排文字' },
  backgroundText:     { en: ' Background text',   zh: ' 背景文字' },
  shapeLabel:         { en: 'Shape',              zh: '形状' },
  shapeRect:          { en: 'Rect',               zh: '矩形' },
  shapeBubble:        { en: 'Bubble',             zh: '椭圆形' },

  // State badges
  stateDetected:      { en: 'detected',           zh: '已检测' },
  stateOcrDone:       { en: 'ocr done',           zh: '已识别' },
  stateTranslated:    { en: 'translated',         zh: '已翻译' },
  stateReviewed:      { en: 'reviewed',           zh: '已审校' },
  stateDraft:         { en: 'draft',              zh: '草稿' },

  // Draft lasso
  draftHint:          { en: 'Incomplete freehand shape — continue drawing or finish.', zh: '未完成的自由形 — 继续绘制或完成。' },
  continueLasso:      { en: 'Continue Drawing',   zh: '继续绘制' },
  finishDraft:        { en: 'Finish Shape',        zh: '完成形状' },

  // Add bubble section
  addBubbleBtn:       { en: 'Add Box/Round/Freehand', zh: '添加方/圆/自由形' },
  boxOption:          { en: 'Box',                zh: '方形' },
  roundOption:        { en: 'Round',              zh: '圆形' },
  frehandOption:      { en: 'Freehand',           zh: '自由形' },

  // Dict panel
  dictTitle:          { en: 'Dictionary',         zh: '词典' },
  dictTabSeg:         { en: 'No-Split',           zh: '禁止分词' },
  dictTabGlossary:    { en: 'Glossary',           zh: '术语表' },
  dictSegPlaceholder: { en: 'Keep phrase whole…', zh: '保持短语完整…' },
  dictJaPlaceholder:  { en: 'JP term',            zh: '日文术语' },
  dictZhPlaceholder:  { en: 'ZH override',        zh: '中文替换' },
  dictResetAll:       { en: 'Reset all',          zh: '全部重置' },
  dictEmptySeg:       { en: 'No entries — add phrases\nBudouX will not split', zh: '暂无条目 — 添加短语\nBudouX 不会将其拆分' },
  dictEmptyGlossary:  { en: 'No entries — add JP→ZH\nterm overrides for the LLM', zh: '暂无条目 — 添加日文→中文\n大语言模型 翻译时会自动转换' },

  // Inpaint color popup
  inpaintColorLabel:  { en: 'Inpaint Color',  zh: '修复颜色' },

  // Settings modal
  settingsTitle:      { en: 'Settings',           zh: '设置' },
  settingsNoAccount:  { en: 'No-Account Mode — Use ChatGPT or DeepSeek Web', zh: '无账号模式 — 使用 ChatGPT 或 DeepSeek 网页版' },
  settingsStep1:      { en: 'Run <strong>OCR All</strong> so every bubble has Japanese text.', zh: '运行<strong>全部 OCR</strong>使每个气泡都有日文文字。' },
  settingsStep2:      { en: 'Click <strong>Copy Prompt</strong> in the toolbar — copies the full translation prompt.', zh: '点击工具栏中的<strong>复制提示词</strong>——复制完整翻译提示。' },
  settingsStep3:      { en: 'Open your AI chat below, paste the prompt, and wait for the response.', zh: '打开下方的 AI 链接，粘贴提示词并等待回复。' },
  settingsStep4:      { en: 'Copy the <em>entire</em> response (the JSON array must be included).', zh: '复制<em>完整</em>回复（必须包含 JSON 数组）。' },
  settingsStep5:      { en: 'Back in Kalar, click <strong>Paste Response</strong> — translations are applied automatically.', zh: '回到 Kalar，点击<strong>粘贴回复</strong>——翻译将自动应用。' },
} satisfies Record<string, { en: string; zh: string }>

export type I18nKey = keyof typeof STRINGS

const LOCALE_KEY = 'kalar_locale'
let _locale: Locale = (localStorage.getItem(LOCALE_KEY) as Locale | null) ?? 'en'
const _listeners: (() => void)[] = []

export function t(key: I18nKey): string {
  return STRINGS[key][_locale]
}

export function setLocale(locale: Locale): void {
  _locale = locale
  localStorage.setItem(LOCALE_KEY, locale)
  _listeners.forEach(fn => fn())
}

export function getLocale(): Locale { return _locale }

export function onLocaleChange(fn: () => void): void {
  _listeners.push(fn)
}

export function applyLocale(root: Element | Document = document): void {
  root.querySelectorAll<HTMLElement>('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n as I18nKey
    if (key in STRINGS) el.textContent = t(key)
  })
  root.querySelectorAll<HTMLElement>('[data-i18n-html]').forEach(el => {
    const key = el.dataset.i18nHtml as I18nKey
    if (key in STRINGS) el.innerHTML = t(key)
  })
  root.querySelectorAll<HTMLInputElement>('[data-i18n-placeholder]').forEach(el => {
    const key = el.dataset.i18nPlaceholder as I18nKey
    if (key in STRINGS) el.placeholder = t(key)
  })
}
