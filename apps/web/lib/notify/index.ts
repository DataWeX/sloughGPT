/**
 * notify — the in-repo notification library.
 *
 * One time source (engine), two topologies (channel: bounded stack / single
 * slot), two instances (toast / banner), one facade:
 *
 *   notify.toast(message, { type, sticky, ttlMs, key, verbose, onUndo })
 *   notify.banner({ tone, title, message, action, key, ttlMs })
 *
 * Store hooks (`useToastStore` / `useBannerStore`) remain for reading state
 * and dismiss/clear actions; announcing goes through `notify`.
 */

import { banner } from './banner'
import { toast } from './toast'

export const notify = { toast, banner }

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
