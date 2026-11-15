/**
 * notify — the in-repo notification library / unified dispatch handle
 * (reconciled 2026-11-14: this directory wins over main's former
 * `lib/notify.ts` dispatcher — same six-intent contract, richer toast and
 * banner options underneath).
 *
 * One time source (engine), two topologies (channel: bounded stack / single
 * slot), two instances (toast / banner), one facade — SIX intents, a thin
 * pure dispatcher with no state, timers or policy of its own:
 *
 *   notify.toast(message, { type, sticky, ttlMs, key, verbose, onUndo })
 *   notify.dismissToast(id) / notify.clearToasts()
 *   notify.banner({ tone, title, message, action, key, ttlMs })
 *   notify.dismissBanner(id)
 *   notify.process(event)   → process-notifications dock projector
 *
 * Tier SELECTION stays a producer decision (AGENTS.md: toasts = transient
 * confirmations, banner = journey blocker with actions). Store hooks
 * (`useToastStore` / `useBannerStore`) remain for READING state; announcing
 * and dismissal go through `notify`.
 *
 * NOT on this surface: `dispatchAction` (the dock's server-validated
 * command execution) — notifications are not commands.
 */

import { ingestProcessEvents, type ProcessEvent } from '@/lib/process-notifications'
import { banner, dismissBanner } from './banner'
import { clearToasts, dismissToast, toast } from './toast'

function process(event: ProcessEvent): void {
  ingestProcessEvents([event])
}

export const notify = { toast, dismissToast, clearToasts, banner, dismissBanner, process }

export { useToastStore, toast, dismissToast, clearToasts, type ToastState } from './toast'
export { useBannerStore, banner, dismissBanner, clearBanners, type BannerState } from './banner'
export { createChannel, type Channel, type ChannelConfig } from './channel'
export {
  MAX_VISIBLE_TOASTS,
  DEFAULT_TTL_MS,
  UNDO_TTL_MS,
  resolveTag,
  makeRoom,
  isTimed,
  isExpired,
  resetEngine,
} from './engine'
export type { NotifyEntry, NotifyPolicy, Topology } from './engine'
export type {
  Toast,
  ToastOptions,
  ToastType,
  Banner,
  BannerInput,
  BannerTone,
  BannerAction,
} from './types'
