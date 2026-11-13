import { formatDateTime } from '@/lib/time-format'

export interface TrainingRun {
  run_id: string
  timestamp: number
  model?: string
  method?: string
  epochs?: number
  final_loss?: number
  best_loss?: number
  perplexity?: number
  quality_score?: number
  training_time_s?: number
  converged?: boolean
  early_stopped?: boolean
  dataset?: string
  tags?: string[]
  notes?: string
  bookmarked?: boolean
}

export function formatDate(ts: number) {
  if (!ts || !Number.isFinite(ts)) return '-'
  return formatDateTime(new Date(ts * 1000)) || '-'
}

export function formatDuration(secs?: number) {
  if (!secs) return '-'
  const m = Math.floor(secs / 60)
  const s = Math.round(secs % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

export function downloadBlob(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
