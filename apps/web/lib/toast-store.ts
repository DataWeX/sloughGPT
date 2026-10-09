'use client'

import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id: string
  message: string
  type: ToastType
  verbose?: string
  onUndo?: () => void
  key?: string
}

interface ToastStore {
  toasts: Toast[]
  addToast: (
    message: string,
    type?: ToastType,
    verbose?: string,
    onUndo?: () => void,
    key?: string,
  ) => string
  dismissToast: (id: string) => void
  clearToasts: () => void
}

const dedupeTag = (message: string, type: ToastType, key?: string) => key ?? `${type}:${message}`

const toastStore = createStore<ToastStore>((set, get) => ({
  toasts: [],

  addToast: (message, type = 'info', verbose, onUndo, key) => {
    const tag = dedupeTag(message, type, key)
    const existing = get().toasts.find((t) => dedupeTag(t.message, t.type, t.key) === tag)
    if (existing) return existing.id
    const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
    const toast: Toast = { id, message, type, verbose, onUndo, key }
    set((prev) => ({ toasts: [...prev.toasts, toast] }))
    setTimeout(
      () => {
        const current = get().toasts
        if (current.find((t) => t.id === id)) {
          set((prev) => ({ toasts: prev.toasts.filter((t) => t.id !== id) }))
        }
      },
      onUndo ? 8000 : 6000,
    )
    return id
  },

  dismissToast: (id) => {
    set((prev) => ({ toasts: prev.toasts.filter((t) => t.id !== id) }))
  },

  clearToasts: () => {
    set({ toasts: [] })
  },
}))

export const useToastStore = Object.assign(
  <T>(selector: (state: ToastStore) => T): T => useStore(toastStore, selector),
  { getState: toastStore.getState },
)
