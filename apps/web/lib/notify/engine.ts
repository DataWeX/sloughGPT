/**
 * notify engine — the pure core of the in-repo notification library.
 *
 * One time source for the whole library: channels register a sweep provider,
 * and a SINGLE ticker runs only while at least one channel holds a timed
 * entry. Expiry is DATA (`dismissAt` on the entry), never a per-entry timer,
 * so a re-announced event refreshes its deadline by rewriting the field, and
 * a superseded entry cannot expire its replacement — the replacement simply
 * carries its own deadline.
 *
 * Pure: no zustand, no React. Topology-aware helpers (`makeRoom`) are shared
 * by every channel; the channels themselves live in `channel.ts`.
 */

export interface NotifyEntry {
  id: string
  /** Sticky entries never auto-expire — only the user dismisses them. */
  sticky?: boolean
  /** Absolute expiry (epoch ms). Absent when sticky. */
  dismissAt?: number
}

export interface NotifyPolicy {
  /** Persist until dismissed instead of auto-expiring. */
  sticky?: boolean
  /** Custom TTL in ms (ignored when `sticky`). */
  ttlMs?: number
}

/** A channel is a topology: a bounded stack (toasts) or a single slot (banners). */
export type Topology = 'stack' | 'slot'

export const SWEEP_INTERVAL_MS = 250
export const DEFAULT_TTL_MS = 6000
export const UNDO_TTL_MS = 8000
export const MAX_VISIBLE_TOASTS = 5

/** Dedupe tag: an explicit key wins; otherwise type+message identify the event. */
export const resolveTag = (message: string, type: string, key?: string): string =>
  key ?? `${type}:${message}`

/** Default persistence: undo affordances get a longer window to read. */
export const defaultTtl = (onUndo?: unknown): number => (onUndo ? UNDO_TTL_MS : DEFAULT_TTL_MS)

export const expiryAt = (now: number, ttlMs: number): number => now + ttlMs

export const isTimed = (entry: NotifyEntry): boolean =>
  !entry.sticky && entry.dismissAt !== undefined

export const isExpired = (entry: NotifyEntry, now: number): boolean =>
  entry.dismissAt !== undefined && entry.dismissAt <= now

/**
 * Make room for ONE incoming stack entry: evict the oldest non-sticky row; if
 * the stack is all-sticky, persistence loses to the bound (oldest goes). Called
 * BEFORE the append so a fresh entry can never evict itself.
 */
export const makeRoom = <T extends NotifyEntry>(list: T[], max: number): T[] => {
  const next = [...list]
  while (next.length >= max) {
    const idx = next.findIndex((t) => !t.sticky)
    next.splice(idx === -1 ? 0 : idx, 1)
  }
  return next
}

/* ── The one ticker ──────────────────────────────────────────── */

export interface SweepProvider {
  /** Remove expired entries from this channel (called only when needs()). */
  sweep(now: number): void
  /** Does this channel currently hold timed entries? */
  needs(): boolean
}

const providers = new Set<SweepProvider>()
let ticker: ReturnType<typeof setInterval> | null = null

const anyNeedsSweep = (): boolean => {
  for (const p of providers) if (p.needs()) return true
  return false
}

export const stopTicker = (): void => {
  if (ticker !== null) {
    clearInterval(ticker)
    ticker = null
  }
}

/** Start the ticker if any channel has timed entries; stop it otherwise. */
export const syncTicker = (): void => {
  if (ticker === null && anyNeedsSweep()) {
    ticker = setInterval(() => {
      const now = Date.now()
      for (const p of providers) if (p.needs()) p.sweep(now)
      if (!anyNeedsSweep()) stopTicker()
    }, SWEEP_INTERVAL_MS)
  } else if (ticker !== null && !anyNeedsSweep()) {
    stopTicker()
  }
}

/** Register a channel's sweep provider; idempotent per object identity. */
export const registerProvider = (p: SweepProvider): void => {
  providers.add(p)
  syncTicker()
}

/** Test hook: drop every provider and stop the ticker. */
export const resetEngine = (): void => {
  stopTicker()
  providers.clear()
}
