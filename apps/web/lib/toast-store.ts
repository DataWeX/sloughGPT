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
}

interface ToastStore {
  toasts: Toast[]
  addToast: (message: string, type?: ToastType, verbose?: string, onUndo?: () => void) => string
  dismissToast: (id: string) => void
  clearToasts: () => void
}

const toastStore = createStore<ToastStore>((set, get) => ({
  toasts: [],

  addToast: (message, type = 'info', verbose, onUndo) => {
    const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
    const toast: Toast = { id, message, type, verbose, onUndo }
    set(prev => ({ toasts: [...prev.toasts, toast] }))
    setTimeout(() => {
      const current = get().toasts
      if (current.find(t => t.id === id)) {
        set(prev => ({ toasts: prev.toasts.filter(t => t.id !== id) }))
      }
    }, onUndo ? 8000 : 6000)
    return id
  },

  dismissToast: (id) => {
    set(prev => ({ toasts: prev.toasts.filter(t => t.id !== id) }))
  },

  clearToasts: () => {
    set({ toasts: [] })
  },
}))

export const useToastStore = Object.assign(
  <T>(selector: (state: ToastStore) => T): T =>
    useStore(toastStore, selector),
  { getState: toastStore.getState },
)
