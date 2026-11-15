'use client'

/**
 * notify() — the unified notification dispatch handle (card 20261107_064201).
 *
 * ONE INTENT API over the three notification tiers. A THIN PURE DISPATCHER:
 * no state, no timers, no policy of its own — it forwards intents verbatim to
 * the tier stores, which stay the state owners:
 *
 *   - toast  → toast-store (transient confirmations; store owns dedupe + TTL)
 *   - banner → banner-store (journey blockers; store owns the one-banner slot)
 *   - process → process-notifications dock projector (structured actions)
 *
 * Producers raise intents here; the stores render. Tier SELECTION stays a
 * producer decision (AGENTS.md: toasts = transient, banner = journey blocker
 * with actions) — this handle adds no auto-routing magic.
 *
 * NOT on this surface: `dispatchAction` (the dock's server-validated command
 * execution). Notifications are not commands — notification intent must never
 * become an execution path.
 */

import { useToastStore } from '@/lib/toast-store'
import { useBannerStore, type Banner } from '@/lib/banner-store'
import { ingestProcessEvents, type ProcessEvent } from '@/lib/process-notifications'

/**
 * Toast intent — mirrors `addToast` argument-for-argument (labeled tuple:
 * names stay visible in the signature, but the caller's arguments are
 * forwarded untouched — no padding with `undefined`, no re-ordering).
 */
function toast(
  ...args: [
    message: string,
    type?: 'success' | 'error' | 'info',
    verbose?: string,
    onUndo?: () => void,
    key?: string,
  ]
): string {
  return useToastStore.getState().addToast(...args)
}

function dismissToast(id: string): void {
  useToastStore.getState().dismissToast(id)
}

function clearToasts(): void {
  useToastStore.getState().clearToasts()
}

/** Banner intent — the store returns the slot id (producers may dismiss later). */
function banner(b: Omit<Banner, 'id'>): string {
  return useBannerStore.getState().showBanner(b)
}

function dismissBanner(id: string): void {
  useBannerStore.getState().dismissBanner(id)
}

/** Process/dock intent — the projector's intake is a batch; single-event sugar. */
function process(event: ProcessEvent): void {
  ingestProcessEvents([event])
}

export const notify = {
  toast,
  dismissToast,
  clearToasts,
  banner,
  dismissBanner,
  process,
} as const
