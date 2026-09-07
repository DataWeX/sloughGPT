/**
 * Duration formatting utilities.
 * 
 * - formatDuration: seconds → "2h 35m" or "5m 30s" or "30s"
 * - formatDurationMs: milliseconds → "1:05:35" or "05:35"
 * - formatDurationCompact: milliseconds → "1.25s" or "350ms"
 */

/** Format seconds to human-readable: "2h 35m", "5m 30s", "30s" */
export function formatDuration(sec: number | null): string {
  if (sec == null || !Number.isFinite(sec) || sec < 0) return '--'
  const total = Math.round(sec)
  const s = total % 60
  const m = Math.floor(total / 60) % 60
  const h = Math.floor(total / 3600)
  if (h > 0) return `${h}h ${m.toString().padStart(2, '0')}m`
  if (m > 0) return `${m}m ${s.toString().padStart(2, '0')}s`
  return `${s}s`
}

/** Format milliseconds to mm:ss or h:mm:ss (audio/video style). */
export function formatDurationMs(ms: number): string {
  if (!ms || ms <= 0) return '0:00'
  const totalSeconds = Math.floor(ms / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
  }
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}

/** Format milliseconds to compact: "1.25s" or "350ms". */
export function formatDurationCompact(ms: number): string {
  if (!ms || ms <= 0) return '0ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)}s`
}

/** Calculate elapsed time between two timestamps and format it. */
export function formatElapsed(start: string | number | Date | null, end?: string | number | Date | null): string {
  if (!start) return ''
  const s = typeof start === 'number' ? start : new Date(start).getTime()
  const e = end ? (typeof end === 'number' ? end : new Date(end).getTime()) : Date.now()
  const ms = e - s
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  const min = Math.floor(ms / 60000)
  const sec = Math.floor((ms % 60000) / 1000)
  return `${min}m ${sec}s`
}
