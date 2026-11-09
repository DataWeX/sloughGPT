/** Public types of the notify library. */

import type { NotifyPolicy } from './engine'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id: string
  message: string
  type: ToastType
  verbose?: string
  onUndo?: () => void
  key?: string
  /** Sticky toasts never auto-expire — pinned until the user dismisses them. */
  sticky?: boolean
  /** Absolute expiry (epoch ms); absent when sticky. */
  dismissAt?: number
}

export interface ToastOptions extends NotifyPolicy {
  /** Which affordance the toast reports. Defaults to `info`. */
  type?: ToastType
  /** Expandable detail row. */
  verbose?: string
  /** Renders an Undo affordance (also widens the default TTL). */
  onUndo?: () => void
  /** Producer dedupe key — re-announcing it refreshes the visible toast. */
  key?: string
}

export type BannerTone = 'info' | 'success' | 'warning' | 'destructive'

export interface BannerAction {
  label: string
  onAction: () => void
}

export interface Banner {
  id: string
  tone: BannerTone
  /** Headline of the banner — short, sentence case. */
  title: string
  /** Supporting detail; rendered muted after the title, em-dash separated. */
  message?: string
  /** Primary remedy — the one thing the user should be able to do next. */
  action?: BannerAction
  /** Producer's dedupe key: informational — the showing banner gets replaced. */
  key?: string
  /** Sticky banners carry no deadline (default). */
  sticky?: boolean
  /** Absolute expiry (epoch ms) when created with `ttlMs`. */
  dismissAt?: number
}

export interface BannerInput extends Omit<Banner, 'id' | 'sticky' | 'dismissAt'> {
  /** Auto-dismiss after this many ms. Omit for the default: sticky. */
  ttlMs?: number
}
