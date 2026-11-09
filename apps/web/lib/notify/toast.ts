/**
 * toast channel — bounded stack of transient confirmations/alerts.
 *
 * `toast()` is the library's public entry: dedupe by tag (a re-announcement
 * refreshes the deadline, never spawns a second row), sticky upgrade on
 * re-announcement, bounded stack via the channel's `push`.
 */

import { createChannel } from './channel'
import { defaultTtl, expiryAt, MAX_VISIBLE_TOASTS, resolveTag } from './engine'
import type { Toast, ToastOptions } from './types'

export interface ToastState {
  toasts: Toast[]
  dismissToast: (id: string) => void
  clearToasts: () => void
}

const channel = createChannel<Toast, ToastState>({
  key: 'toasts',
  topology: 'stack',
  max: MAX_VISIBLE_TOASTS,
  dismissName: 'dismissToast',
  clearName: 'clearToasts',
})

/** Announce a toast. Returns the visible toast's id. */
export function toast(message: string, opts: ToastOptions = {}): string {
  const { type = 'info', verbose, onUndo, key, sticky, ttlMs } = opts

  const tag = resolveTag(message, type, key)
  const existing = channel.list().find((t) => resolveTag(t.message, t.type, t.key) === tag)
  if (existing) {
    // Re-announcement: refresh the deadline (a live event must not die
    // mid-importance) and upgrade to sticky if the new call asks for it
    // (sticky never downgrades back).
    const nextSticky = existing.sticky || sticky
    const ttl = ttlMs ?? defaultTtl(existing.onUndo)
    channel.patch(existing.id, {
      sticky: nextSticky,
      dismissAt: nextSticky ? undefined : expiryAt(Date.now(), ttl),
    })
    return existing.id
  }

  const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
  channel.push({
    id,
    message,
    type,
    verbose,
    onUndo,
    key,
    sticky,
    dismissAt: sticky ? undefined : expiryAt(Date.now(), ttlMs ?? defaultTtl(onUndo)),
  })
  return id
}

export const useToastStore = Object.assign(channel.useChannel, { getState: channel.getState })
export const dismissToast = channel.remove
export const clearToasts = channel.clear
