'use client'

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '../../lib/cn'
import { cva, type VariantProps } from 'class-variance-authority'

/* ── Types ──────────────────────────────────────────────────── */

export type ToastVariant = 'default' | 'success' | 'error' | 'warning' | 'info'

export interface ToastOptions {
  id?: string
  title: string
  description?: string
  variant?: ToastVariant
  duration?: number
  action?: {
    label: string
    onClick: () => void
  }
}

interface ToastState extends Required<Pick<ToastOptions, 'id' | 'title' | 'variant' | 'duration'>> {
  description?: string
  action?: ToastOptions['action']
  visible: boolean
}

/* ── Context ────────────────────────────────────────────────── */

interface ToastContextValue {
  toasts: ToastState[]
  toast: (options: ToastOptions) => string
  dismiss: (id: string) => void
  dismissAll: () => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

/* ── Hook ───────────────────────────────────────────────────── */

export function useToast() {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within <ToastProvider>')
  return ctx
}

/* ── Provider ───────────────────────────────────────────────── */

export function ToastProvider({ children, maxToasts = 5 }: { children: ReactNode; maxToasts?: number }) {
  const [toasts, setToasts] = useState<ToastState[]>([])

  const dismiss = useCallback((id: string) => {
    setToasts((prev) =>
      prev.map((t) => (t.id === id ? { ...t, visible: false } : t))
    )
    // Remove from DOM after animation
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id))
    }, 350)
  }, [])

  const dismissAll = useCallback(() => {
    setToasts((prev) => prev.map((t) => ({ ...t, visible: false })))
    setTimeout(() => setToasts([]), 350)
  }, [])

  const toast = useCallback(
    (options: ToastOptions): string => {
      const id = options.id ?? `toast-${Date.now()}-${Math.random().toString(36).slice(2)}`
      const duration = options.duration ?? 5000

      const newToast: ToastState = {
        id,
        title: options.title,
        description: options.description,
        variant: options.variant ?? 'default',
        duration,
        action: options.action,
        visible: true,
      }

      setToasts((prev) => {
        const next = [newToast, ...prev]
        return next.slice(0, maxToasts)
      })

      if (duration > 0) {
        setTimeout(() => dismiss(id), duration)
      }

      return id
    },
    [dismiss, maxToasts]
  )

  const value = useMemo(() => ({ toasts, toast, dismiss, dismissAll }), [toasts, toast, dismiss, dismissAll])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <ToastViewport />
    </ToastContext.Provider>
  )
}

/* ── Variant styles ─────────────────────────────────────────── */

const toastVariants = cva(
  [
    'pointer-events-auto relative flex w-full items-start gap-3 overflow-hidden rounded-lg border p-4 shadow-lg',
    'transition-all duration-300',
  ].join(' '),
  {
    variants: {
      variant: {
        default: 'bg-background border-border text-foreground',
        success: 'bg-success/10 border-success/30 text-foreground',
        error: 'bg-destructive/10 border-destructive/30 text-foreground',
        warning: 'bg-warning/10 border-warning/30 text-foreground',
        info: 'bg-accent border-accent-foreground/10 text-foreground',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

/* ── Icons ──────────────────────────────────────────────────── */

function ToastIcon({ variant }: { variant: ToastVariant }) {
  if (variant === 'success') {
    return (
      <svg className="mt-0.5 h-4 w-4 shrink-0 text-success" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  }
  if (variant === 'error') {
    return (
      <svg className="mt-0.5 h-4 w-4 shrink-0 text-destructive" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  }
  if (variant === 'warning') {
    return (
      <svg className="mt-0.5 h-4 w-4 shrink-0 text-warning" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
      </svg>
    )
  }
  if (variant === 'info') {
    return (
      <svg className="mt-0.5 h-4 w-4 shrink-0 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  }
  return null
}

/* ── Toast item ─────────────────────────────────────────────── */

function ToastItem({ toast: t, onDismiss }: { toast: ToastState; onDismiss: (id: string) => void }) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-atomic="true"
      className={cn(
        toastVariants({ variant: t.variant }),
        t.visible ? 'translate-y-0 opacity-100' : 'translate-y-2 opacity-0',
      )}
    >
      <ToastIcon variant={t.variant} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold leading-snug">{t.title}</p>
        {t.description && (
          <p className="mt-0.5 text-xs text-muted-foreground leading-relaxed">{t.description}</p>
        )}
        {t.action && (
          <button
            type="button"
            onClick={() => { t.action!.onClick(); onDismiss(t.id) }}
            className="mt-1.5 text-xs font-medium text-primary hover:underline focus:outline-none"
          >
            {t.action.label}
          </button>
        )}
      </div>
      <button
        type="button"
        aria-label="Dismiss"
        onClick={() => onDismiss(t.id)}
        className="ml-auto flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-primary/40"
      >
        <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  )
}

/* ── Viewport ───────────────────────────────────────────────── */

function ToastViewport() {
  const ctx = useContext(ToastContext)
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  if (!mounted || !ctx || ctx.toasts.length === 0) return null

  return createPortal(
    <div
      aria-label="Notifications"
      className="fixed bottom-0 right-0 z-[100] flex flex-col-reverse gap-2 p-4 max-h-screen w-full sm:w-[400px] pointer-events-none"
    >
      {ctx.toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={ctx.dismiss} />
      ))}
    </div>,
    document.body,
  )
}

/* ── Primitive exports (backwards-compat) ───────────────────── */

export {
  ToastViewport,
  toastVariants,
}

// Primitive shims for backwards compatibility with existing imports
export function Toast({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof toastVariants>) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={cn(toastVariants({ variant: props.variant }), className)}
      {...props}
    >
      {children}
    </div>
  )
}

export function ToastTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h2 className={cn('text-sm font-semibold', className)} {...props} />
}

export function ToastDescription({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn('text-xs text-muted-foreground', className)} {...props} />
}

export function ToastClose({ className, onClick, ...props }: React.HTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      aria-label="Close"
      className={cn(
        'ml-auto flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-primary/40',
        className,
      )}
      onClick={onClick}
      {...props}
    >
      <svg className="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden>
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  )
}

export function ToastAction({ className, ...props }: React.HTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cn(
        'inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-xs font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:pointer-events-none disabled:opacity-50',
        className,
      )}
      {...props}
    />
  )
}

import * as React from 'react'
