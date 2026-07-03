/**
 * Whether a catalog model id corresponds to the API's currently loaded `model_type`
 * from `GET /health` (ids may be short names or path-like).
 */
export function catalogIdMatchesRuntime(catalogId: string, runtimeType: string): boolean {
  const c = catalogId.trim().toLowerCase()
  const r = runtimeType.trim().toLowerCase()
  if (!c || !r) return false
  if (c === r) return true
  const cLast = c.split('/').pop() ?? c
  const rLast = r.split('/').pop() ?? r
  if (cLast === rLast) return true
  if (c.endsWith(`/${r}`) || c.endsWith(`/${rLast}`)) return true
  if (r.endsWith(`/${c}`)) return true
  return false
}

/**
 * Generate a human-friendly display name from a HuggingFace model ID.
 * Used as a fallback when the backend `name` field is not available.
 *
 * Examples:
 *   "Qwen/Qwen2.5-0.5B-Instruct" → "Qwen 2.5 0.5B Instruct"
 *   "gpt2" → "GPT-2"
 *   "microsoft/Phi-3.5-mini-instruct" → "Phi 3.5 Mini Instruct"
 */
const _DISPLAY_NAMES: Record<string, string> = {
  'gpt2': 'GPT-2',
  'gpt2-medium': 'GPT-2 Medium',
  'gpt2-large': 'GPT-2 Large',
  'gpt2-xl': 'GPT-2 XL',
}

export function modelDisplayName(modelId: string): string {
  if (_DISPLAY_NAMES[modelId]) return _DISPLAY_NAMES[modelId]
  const name = modelId.includes('/') ? modelId.split('/').pop()! : modelId
  return name.replace(/[-_]/g, ' ').replace(/\s+/g, ' ').trim()
}
