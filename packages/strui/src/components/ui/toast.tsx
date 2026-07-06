'use client'

import {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type HTMLAttributes,
  type ReactNode,
} from 'react'
import { createPortal } from 'react-dom'
import { cn } from '@/lib/cn'

/* ── Toast Provider (context only, no visual) ───────────────────── */

interface ToastProviderContextValue {
  toasts: Set<string>
  addToast: (id: string) => void
  removeToast: (id: string) => void
}

const ToastProviderContext = createContext<ToastProviderContextValue | null>(null)

function ToastProvider({ children, duration }: { children: ReactNode; duration?: number }) {
  const [toasts, setToasts] = useState<Set<string>>(new Set())

  const addToast = useCallback((id: string) => setToasts((prev) => new Set(prev).add(id)), [])
  const removeToast = useCallback((id: string) =>
    setToasts((prev) => {
      const next = new Set(prev)
      next.delete(id)
      return next
    }), [])

  const value = useMemo(() => ({ toasts, addToast, removeToast }), [toasts, addToast, removeToast])

  return (
    <ToastProviderContext.Provider value={value}>
      {children}
    </ToastProviderContext.Provider>
  )
}

/* ── Toast Viewport ─────────────────────────────────────────────── */

function ToastViewport({ className }: { className?: string }) {
  if (typeof document === 'undefined') return null
  return createPortal(
    <div
      aria-live="polite"
      className={cn(
        'fixed bottom-0 right-0 z-[100] flex flex-col gap-2 p-4 max-h-screen w-full sm:w-[380px] pointer-events-none',
        className,
      )}
    />,
    document.body,
  )
}

/* ── Toast Root ─────────────────────────────────────────────────── */

interface ToastProps extends HTMLAttributes<HTMLDivElement> {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  duration?: number
}

const Toast = forwardRef<HTMLDivElement, ToastProps>(
  ({ className, open: controlledOpen, onOpenChange, duration = 5000, children, ...props }, ref) => {
    const ctx = useContext(ToastProviderContext)
    const [internalOpen, setInternalOpen] = useState(false)
    const isControlled = controlledOpen !== undefined
    const open = isControlled ? controlledOpen : internalOpen
    const id = useState(() => Math.random().toString(36).slice(2))[0]
    const ctxRef = useRef(ctx)
    ctxRef.current = ctx

    useEffect(() => {
      if (!isControlled) setInternalOpen(controlledOpen ?? false)
    }, [controlledOpen, isControlled])

    useEffect(() => {
      const c = ctxRef.current
      if (open) {
        c?.addToast(id)
        if (duration > 0) {
          const timer = setTimeout(() => {
            onOpenChange?.(false)
            if (!isControlled) setInternalOpen(false)
          }, duration)
          return () => clearTimeout(timer)
        }
      } else {
        c?.removeToast(id)
      }
    }, [open, duration, onOpenChange, isControlled, id])

    if (!open) return null

    return (
      <div
        ref={ref}
        role="status"
        aria-live="polite"
        className={cn(
          'pointer-events-auto w-full rounded-lg border bg-background p-4 shadow-lg',
          'animate-in slide-in-from-bottom-5 fade-in-0 duration-300',
          className,
        )}
        {...props}
      >
        {children}
      </div>
    )
  },
)
Toast.displayName = 'Toast'

/* ── Toast Title ────────────────────────────────────────────────── */

const ToastTitle = forwardRef<HTMLHeadingElement, HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h2 ref={ref} className={cn('text-sm font-semibold', className)} {...props} />
  ),
)
ToastTitle.displayName = 'ToastTitle'

/* ── Toast Description ──────────────────────────────────────────── */

const ToastDescription = forwardRef<HTMLParagraphElement, HTMLAttributes<HTMLParagraphElement>>(
  ({ className, ...props }, ref) => (
    <p ref={ref} className={cn('text-sm text-muted-foreground', className)} {...props} />
  ),
)
ToastDescription.displayName = 'ToastDescription'

/* ── Toast Close ────────────────────────────────────────────────── */

const ToastClose = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ className, onClick, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      aria-label="Close"
      className={cn(
        'absolute right-2 top-2 rounded-md p-1 text-foreground/50 opacity-0 transition-opacity hover:text-foreground focus:opacity-100 focus:outline-none',
        'group-hover:opacity-100',
        className,
      )}
      onClick={(e) => {
        onClick?.(e)
        const parent = (e.target as HTMLElement).closest('[role="status"]')
        if (parent) parent.remove()
      }}
      {...props}
    >
      <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  ),
)
ToastClose.displayName = 'ToastClose'

/* ── Toast Action ───────────────────────────────────────────────── */

const ToastAction = forwardRef<HTMLButtonElement, HTMLAttributes<HTMLButtonElement>>(
  ({ className, ...props }, ref) => (
    <button
      ref={ref}
      type="button"
      className={cn(
        'inline-flex h-8 shrink-0 items-center justify-center rounded-md border bg-transparent px-3 text-sm font-medium transition-colors hover:bg-secondary focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:pointer-events-none disabled:opacity-50',
        className,
      )}
      {...props}
    />
  ),
)
ToastAction.displayName = 'ToastAction'

export { ToastProvider, ToastViewport, Toast, ToastTitle, ToastDescription, ToastClose, ToastAction }
