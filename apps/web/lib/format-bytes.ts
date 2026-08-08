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

/** Default max tokens for generation. */
export const DEFAULT_MAX_TOKENS = 100

/** Safely read a JSON value from localStorage, returning fallback on any error. */
export function getJsonItem<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    if (raw === null) return fallback
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}
