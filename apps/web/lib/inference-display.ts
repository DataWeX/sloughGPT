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
 * Fully algorithmic — no hardcoded lookup tables.
 *
 * Examples:
 *   "Qwen/Qwen2.5-0.5B-Instruct" → "Qwen 2.5 0.5B Instruct"
 *   "gpt2" → "GPT 2"
 *   "gpt2-medium" → "GPT 2 Medium"
 *   "gpt2-xl" → "GPT 2 XL"
 *   "microsoft/Phi-3.5-mini-instruct" → "Phi 3.5 Mini Instruct"
 */
export function modelDisplayName(modelId: string): string {
  let name = modelId.includes('/') ? modelId.split('/').pop()! : modelId

  // Strip cache prefix: "models--org--model" → "model"
  if (name.startsWith('models--')) {
    const after = name.slice(8)
    const idx = after.indexOf('--')
    if (idx >= 0) name = after.slice(idx + 2)
  }

  // Split on common separators
  const parts = name.split(/[/\-_]/)

  const result: string[] = []
  for (const part of parts) {
    if (!part) continue
    // Short all-lowercase abbreviations (xl, bp, etc.) → uppercase all
    if (part.length <= 3 && /^[a-z]+$/.test(part)) {
      result.push(part.toUpperCase())
    }
    // All lowercase letters followed by digits: "gpt2", "llama3"
    else if (/^[a-z]+\d+$/.test(part)) {
      const m = part.match(/^([a-z]+)(\d+)$/)
      if (m) { result.push(m[1].toUpperCase()); result.push(m[2]) }
      else result.push(part.toUpperCase())
    }
    // Number with size suffix: "0.5B", "3B", "8B"
    else if (/^\d+\.?\d*[a-zA-Z]$/.test(part)) {
      result.push(part)
    }
    // Normal mixed case
    else {
      let sub = part.replace(/([a-zA-Z]{2,})(\d)/g, '$1 $2')
      sub = sub.replace(/(\d)([a-zA-Z]{2,})/g, '$1 $2')
      for (const w of sub.split(' ')) {
        if (/^[\d.]+$/.test(w)) result.push(w)
        else result.push(w.charAt(0).toUpperCase() + w.slice(1))
      }
    }
  }
  return result.join(' ')
}
