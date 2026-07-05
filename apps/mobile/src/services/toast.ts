/**
 * Lightweight toast notification system for React Native.
 * Non-blocking, auto-dismissing notifications that appear at the top of the screen.
 *
 * Usage:
 *   import {toast} from '../services/toast';
 *   toast.success('Message sent');
 *   toast.error('Connection lost');
 *   toast.info('Syncing...');
 *   toast.warn('Slow connection');
 */

import {triggerHaptic} from './haptics';

export type ToastType = 'success' | 'error' | 'info' | 'warn';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
  timestamp: number;
}

type ToastListener = (toasts: Toast[]) => void;

const listeners: Set<ToastListener> = new Set();
let toasts: Toast[] = [];
let idCounter = 0;

const DURATIONS: Record<ToastType, number> = {
  success: 2500,
  error: 4000,
  info: 2000,
  warn: 3500,
};

function notify() {
  listeners.forEach(fn => fn([...toasts]));
}

function add(type: ToastType, message: string, duration?: number): string {
  const id = `toast-${++idCounter}`;
  const toast: Toast = {
    id,
    type,
    message,
    duration: duration ?? DURATIONS[type],
    timestamp: Date.now(),
  };

  toasts = [toast, ...toasts].slice(0, 5); // max 5 visible
  notify();

  // Haptic feedback for errors
  if (type === 'error') {
    triggerHaptic('error');
  }

  // Auto-dismiss
  setTimeout(() => {
    dismiss(id);
  }, toast.duration);

  return id;
}

function dismiss(id: string) {
  toasts = toasts.filter(t => t.id !== id);
  notify();
}

function clear() {
  toasts = [];
  notify();
}

export const toast = {
  success: (msg: string, duration?: number) => add('success', msg, duration),
  error: (msg: string, duration?: number) => add('error', msg, duration),
  info: (msg: string, duration?: number) => add('info', msg, duration),
  warn: (msg: string, duration?: number) => add('warn', msg, duration),
  dismiss,
  clear,
  /** Subscribe to toast changes. Returns unsubscribe function. */
  subscribe: (fn: ToastListener): (() => void) => {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },
  /** Get current toasts (for initial render). */
  getToasts: () => [...toasts],
};
