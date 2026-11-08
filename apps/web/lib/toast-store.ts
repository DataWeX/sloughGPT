'use client'

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'

export type ToastType = 'success' | 'error' | 'info'

/** Bounded stack: a burst can never cover the viewport. Eviction drops the
 * oldest non-sticky row first; if every row is sticky, persistence loses to
 * the bound (oldest sticky goes) so the stack can never grow unbounded. */
export const MAX_VISIBLE_TOASTS = 5

const SWEEP_INTERVAL_MS = 250
const DEFAULT_TTL_MS = 6000
const UNDO_TTL_MS = 8000

export interface Toast {
  id: string
  message: string
  type: ToastType
  verbose?: string
  onUndo?: () => void
  key?: string
  /** Sticky toasts never auto-expire — only the user dismisses them. */
  sticky?: boolean
  /** Absolute expiry (epoch ms). Absent when sticky; swept centrally. */
  dismissAt?: number
}

export interface ToastOptions {
  /** Persist until dismissed instead of auto-expiring. */
  sticky?: boolean
  /** Custom TTL in ms (ignored when `sticky`). Defaults: 6000, 8000 with undo. */
  ttlMs?: number
}

interface ToastStore {
  toasts: Toast[]
  addToast: (
    message: string,
    type?: ToastType,
    verbose?: string,
    onUndo?: () => void,
    key?: string,
    opts?: ToastOptions,
  ) => string
  dismissToast: (id: string) => void
  clearToasts: () => void
}

const dedupeTag = (message: string, type: ToastType, key?: string) => key ?? `${type}:${message}`

const defaultTtl = (onUndo?: () => void) => (onUndo ? UNDO_TTL_MS : DEFAULT_TTL_MS)

const toastStore = createStore<ToastStore>((set, get) => {
  // One sweeper for the whole store — expiry is DATA (`dismissAt`), not a
  // per-toast timer. Producers pick the policy (sticky / ttlMs); a dedupe
  // hit refreshes the deadline so a re-announced event cannot die mid-importance.
  let sweeper: ReturnType<typeof setInterval> | null = null

  const hasTimedToasts = () => get().toasts.some((t) => !t.sticky && t.dismissAt !== undefined)

  const stopSweeper = () => {
    if (sweeper !== null) {
      clearInterval(sweeper)
      sweeper = null
    }
  }

  const ensureSweeper = () => {
    if (sweeper !== null || !hasTimedToasts()) return
    sweeper = setInterval(() => {
      const now = Date.now()
      if (get().toasts.some((t) => t.dismissAt !== undefined && t.dismissAt <= now)) {
        set((prev) => ({
          toasts: prev.toasts.filter((t) => !(t.dismissAt !== undefined && t.dismissAt <= now)),
        }))
      }
      if (!hasTimedToasts()) stopSweeper()
    }, SWEEP_INTERVAL_MS)
  }

  // Make room for ONE incoming toast: evict the oldest non-sticky row; if the
  // stack is all-sticky, persistence loses to the bound (oldest goes). Called
  // before the append so a fresh toast can never evict itself.
  const makeRoom = (list: Toast[]): Toast[] => {
    const next = [...list]
    while (next.length >= MAX_VISIBLE_TOASTS) {
      const idx = next.findIndex((t) => !t.sticky)
      next.splice(idx === -1 ? 0 : idx, 1)
    }
    return next
  }

  return {
    toasts: [],

    addToast: (message, type = 'info', verbose, onUndo, key, opts) => {
      const tag = dedupeTag(message, type, key)
      const existing = get().toasts.find((t) => dedupeTag(t.message, t.type, t.key) === tag)
      if (existing) {
        // Re-announcement: refresh the deadline and upgrade to sticky if the
        // new call asks for it (sticky never downgrades back).
        const sticky = existing.sticky || opts?.sticky
        const ttl = opts?.ttlMs ?? defaultTtl(existing.onUndo)
        set((prev) => ({
          toasts: prev.toasts.map((t) =>
            t.id === existing.id
              ? { ...t, sticky, dismissAt: sticky ? undefined : Date.now() + ttl }
              : t,
          ),
        }))
        ensureSweeper()
        return existing.id
      }

      const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
      const sticky = opts?.sticky ?? false
      const toast: Toast = {
        id,
        message,
        type,
        verbose,
        onUndo,
        key,
        sticky,
        dismissAt: sticky ? undefined : Date.now() + (opts?.ttlMs ?? defaultTtl(onUndo)),
      }
      set((prev) => ({ toasts: [...makeRoom(prev.toasts), toast] }))
      ensureSweeper()
      return id
    },

    dismissToast: (id) => {
      set((prev) => ({ toasts: prev.toasts.filter((t) => t.id !== id) }))
      if (!hasTimedToasts()) stopSweeper()
    },

    clearToasts: () => {
      set({ toasts: [] })
      stopSweeper()
    },
  }
})

export const useToastStore = Object.assign(
  <T>(selector: (state: ToastStore) => T): T => useStore(toastStore, selector),
  { getState: toastStore.getState },
)
