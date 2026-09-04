'use client'

import { memo, useEffect, useRef, useState } from 'react'
import { cn, IconX, IconCheckCircle, IconError, IconInfo } from '@sloughgpt/strui'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id: string
  message: string
  type: ToastType
  verbose?: string
  onUndo?: () => void
}

interface ToastItemProps {
  toast: Toast
  onDismiss: (id: string) => void
}

const TYPE_STYLES: Record<ToastType, { bar: string; icon: string }> = {
  success: { bar: 'bg-success', icon: 'text-success' },
  error: { bar: 'bg-destructive', icon: 'text-destructive' },
  info: { bar: 'bg-primary', icon: 'text-primary' },
}

const DURATION = 4000
const EXIT_MS = 250

function ToastIcon({ type, className }: { type: ToastType; className?: string }) {
  return (
    <span className={cn('text-sm', className)}>
      {type === 'success' ? <IconCheckCircle className="h-4 w-4 text-success" /> :
       type === 'error' ? <IconError className="h-4 w-4 text-destructive" /> :
       <IconInfo className="h-4 w-4 text-primary" />}
    </span>
  )
}

function ToastItem({ toast, onDismiss }: ToastItemProps) {
  const styles = TYPE_STYLES[toast.type]
  const [exiting, setExiting] = useState(false)
  const [showVerbose, setShowVerbose] = useState(false)
  const [progress, setProgress] = useState(100)

  const startRef = useRef(Date.now())
  const pausedAtRef = useRef(0)
  const isPausedRef = useRef(false)
  const frameRef = useRef<number>(0)
  const dismissedRef = useRef(false)

  const scheduleDismiss = (id: string) => {
    if (dismissedRef.current) return
    dismissedRef.current = true
    setExiting(true)
    setTimeout(() => onDismiss(id), EXIT_MS)
  }

  useEffect(() => {
    startRef.current = Date.now()
    const id = toast.id

    const tick = () => {
      if (isPausedRef.current) {
        frameRef.current = requestAnimationFrame(tick)
        return
      }
      const elapsed = Date.now() - startRef.current
      const remaining = Math.max(0, 100 - (elapsed / DURATION) * 100)
      setProgress(remaining)
      if (remaining <= 0) {
        scheduleDismiss(id)
      } else {
        frameRef.current = requestAnimationFrame(tick)
      }
    }
    frameRef.current = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(frameRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [toast.id])

  const handleMouseEnter = () => {
    if (dismissedRef.current) return
    isPausedRef.current = true
    pausedAtRef.current = Date.now()
  }

  const handleMouseLeave = () => {
    if (!isPausedRef.current) return
    isPausedRef.current = false
    const pauseDuration = Date.now() - pausedAtRef.current
    startRef.current += pauseDuration
  }

  const dismiss = () => {
    scheduleDismiss(toast.id)
  }

  return (
    <div
      className={cn('sl-toast', exiting ? ' sl-toast--exit' : '')}
      role="alert"
      aria-live={toast.type === 'error' ? 'assertive' : 'polite'}
      aria-atomic="true"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onClick={dismiss}
    >
      <span className="sr-only">
        {toast.type === 'success' && 'Success: '}
        {toast.type === 'error' && 'Error: '}
        {toast.type === 'info' && 'Information: '}
      </span>

      <div className={cn('sl-toast__bar', styles.bar)} />

      <span className="sl-toast__body">
        <ToastIcon type={toast.type} className={cn('sl-toast__icon', styles.icon)} />
        <span className="sl-toast__message">
          {toast.message}
          {toast.verbose && (
            <button
              type="button"
              onClick={e => { e.stopPropagation(); setShowVerbose(v => !v) }}
              className="sl-toast__verbose"
            >
              {showVerbose ? 'Hide' : 'Details'}
            </button>
          )}
        </span>
        {toast.onUndo && (
          <button
            type="button"
            onClick={e => { e.stopPropagation(); toast.onUndo!(); dismiss() }}
            className="sl-toast__undo"
          >
            Undo
          </button>
        )}
        <button
          type="button"
          onClick={e => { e.stopPropagation(); dismiss() }}
          className="sl-toast__close"
          aria-label={`Dismiss notification: ${toast.message}`}
        >
          <IconX className="h-3 w-3" />
        </button>
      </span>

      {showVerbose && toast.verbose && (
        <pre className="sl-toast__verbose-block">{toast.verbose}</pre>
      )}

      <span className="sl-toast__track">
        <span
          className={cn('sl-toast__fill', styles.bar)}
          style={{ width: `${progress}%` }}
        />
      </span>
    </div>
  )
}

interface ToastContainerProps {
  toasts: Toast[]
  onDismiss: (id: string) => void
  onClearAll?: () => void
}

export const ToastContainer = memo(function ToastContainer({ toasts, onDismiss, onClearAll }: ToastContainerProps) {
  if (toasts.length === 0) return null
  return (
    <div className="sl-toast-container" role="region" aria-label="Notifications">
      {toasts.length > 2 && onClearAll && (
        <button
          type="button"
          onClick={onClearAll}
          className="absolute top-1 right-1 z-10 text-[9px] text-muted-foreground/50 hover:text-foreground/80 px-1.5 py-0.5 rounded bg-background/60 backdrop-blur-sm transition-colors"
          aria-label="Dismiss all notifications"
        >
          Clear all
        </button>
      )}
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  )
})

// ── Radix Toast Bridge ──────────────────────────────────────────

import {
  ToastProvider, ToastViewport, Toast,
  ToastTitle, ToastDescription, ToastClose,
} from '@sloughgpt/strui'

const TYPE_BORDER: Record<ToastType, string> = {
  success: 'border-l-success',
  error: 'border-l-destructive',
  info: 'border-l-primary',
}

export function RadixToastContainer({ toasts, onDismiss, onClearAll }: ToastContainerProps) {
  return (
    <ToastProvider>
      {toasts.length > 2 && onClearAll && (
        <button
          type="button"
          onClick={onClearAll}
          className="fixed bottom-20 right-4 z-[301] text-[9px] text-muted-foreground/50 hover:text-foreground/80 px-1.5 py-0.5 rounded bg-background/60 backdrop-blur-sm transition-colors"
          aria-label="Dismiss all notifications"
        >
          Clear all
        </button>
      )}
      {toasts.map((t) => (
        <Toast key={t.id} className={['border-l-4', TYPE_BORDER[t.type]].join(' ')}>
          <div className="flex items-start gap-3 w-full">
            <span className="mt-0.5 text-sm shrink-0">
              {t.type === 'success' ? <IconCheckCircle className="h-4 w-4 text-success" /> : t.type === 'error' ? <IconError className="h-4 w-4 text-destructive" /> : <IconInfo className="h-4 w-4 text-primary" />}
            </span>
            <div className="flex-1 min-w-0">
              <ToastTitle className="text-sm">{t.message}</ToastTitle>
              {t.verbose && <ToastDescription className="mt-1">{t.verbose}</ToastDescription>}
            </div>
            <ToastClose onClick={() => onDismiss(t.id)} />
          </div>
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  )
}
