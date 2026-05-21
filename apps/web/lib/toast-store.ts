'use client'

import { create } from 'zustand'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id: string
  message: string
  type: ToastType
  verbose?: string
}

interface ToastStore {
  toasts: Toast[]
  addToast: (message: string, type?: ToastType, verbose?: string) => string
  dismissToast: (id: string) => void
  clearToasts: () => void
}

export const useToastStore = create<ToastStore>((set, get) => ({
  toasts: [],

  addToast: (message, type = 'info', verbose) => {
    const id = `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
    const toast: Toast = { id, message, type, verbose }
    set(prev => ({ toasts: [...prev.toasts, toast] }))
    setTimeout(() => {
      const current = get().toasts
      if (current.find(t => t.id === id)) {
        set(prev => ({ toasts: prev.toasts.filter(t => t.id !== id) }))
      }
    }, 6000)
    return id
  },

  dismissToast: (id) => {
    set(prev => ({ toasts: prev.toasts.filter(t => t.id !== id) }))
  },

  clearToasts: () => {
    set({ toasts: [] })
  },
}))
