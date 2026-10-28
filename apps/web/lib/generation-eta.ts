export interface GenerationEtaInput {
  tokens: number
  elapsedMs: number
  maxTokens?: number | null
}

export interface GenerationEta {
  tokensPerSec: number
  etaMs: number | null
}

export function computeGenerationEta(input: GenerationEtaInput): GenerationEta | null {
  const { tokens, elapsedMs, maxTokens } = input
  if (!(elapsedMs > 0) || !(tokens > 0)) return null
  const tokensPerSec = tokens / (elapsedMs / 1000)
  if (!(tokensPerSec > 0)) return null
  const etaMs =
    maxTokens != null && maxTokens > 0 && tokens < maxTokens
      ? ((maxTokens - tokens) / tokensPerSec) * 1000
      : null
  return { tokensPerSec, etaMs }
}

export function formatEtaDuration(ms: number): string {
  const totalSecs = ms / 1000
  if (totalSecs < 90) return `${Math.max(1, Math.round(totalSecs))}s`
  const mins = Math.floor(totalSecs / 60)
  const secs = Math.round(totalSecs % 60)
  return `${mins}m ${String(secs).padStart(2, '0')}s`
}
