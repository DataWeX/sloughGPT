/** Format byte count to human-readable string (e.g. "1.5 MB"). */
export function formatBytes(bytes: number): string {
  if (!bytes) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

/** Time constants in milliseconds. */
export const MS_PER_SECOND = 1000
export const MS_PER_MINUTE = 60 * MS_PER_SECOND
export const MS_PER_HOUR = 60 * MS_PER_MINUTE
export const MS_PER_DAY = 24 * MS_PER_HOUR

/**
 * Human-readable relative time from a unix-seconds timestamp.
 * "Just now", "5m ago", "3h ago", "2d ago", falling back to a short date
 * once the event is more than a week old.
 */
export function formatRelativeTime(seconds: number): string {
  const ts = seconds * MS_PER_SECOND
  if (!seconds || seconds <= 0 || isNaN(ts)) return ''
  const diffMs = Date.now() - ts
  if (diffMs < MS_PER_MINUTE) return 'Just now'
  if (diffMs < MS_PER_HOUR) return `${Math.floor(diffMs / MS_PER_MINUTE)}m ago`
  if (diffMs < MS_PER_DAY) return `${Math.floor(diffMs / MS_PER_HOUR)}h ago`
  if (diffMs < 7 * MS_PER_DAY) return `${Math.floor(diffMs / MS_PER_DAY)}d ago`
  return new Date(ts).toLocaleDateString()
}

/** Today's date as YYYY-MM-DD for filenames. */
export function todayDateString(): string {
  return new Date().toISOString().slice(0, 10)
}

/** Current timestamp as ISO 8601 string. */
export function nowISO(): string {
  return new Date().toISOString()
}

/** Default error message prefix for toast notifications. */
export const DEFAULT_ERROR_MESSAGE = 'Something went wrong'

/** Default max tokens for PDF analysis. */
export const PDF_ANALYSIS_MAX_TOKENS = 512
