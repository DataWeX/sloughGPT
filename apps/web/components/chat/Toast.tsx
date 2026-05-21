'use client'

import { useEffect, useRef, useState } from 'react'
import { IconX } from '@/components/ui'
import { cn } from '@/lib/cn'

export type ToastType = 'success' | 'error' | 'info'

export interface Toast {
  id: string
  message: string
  type: ToastType
  verbose?: string
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

const successEmojis = ['✨', '🎉', '⭐', '👏', '💪', '🌟', '🎯', '✅']

function ToastIcon({ type, className }: { type: ToastType; className?: string }) {
  const [emoji] = useState(() => successEmojis[Math.floor(Math.random() * successEmojis.length)])

  if (type === 'success') {
    return <span className={cn('text-sm', className)}>{emoji}</span>
  }
  if (type === 'error') {
    return <span className={cn('text-sm', className)}>😕</span>
  }
  return <span className={cn('text-sm', className)}>💡</span>
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
      className={`sl-toast${exiting ? ' sl-toast--exit' : ''}`}
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

      <div className={`sl-toast__bar ${styles.bar}`} />

      <span className="sl-toast__body">
        <ToastIcon type={toast.type} className={`sl-toast__icon ${styles.icon}`} />
        <span className="sl-toast__message">
          {toast.message}
          {toast.verbose && (
            <button
              onClick={e => { e.stopPropagation(); setShowVerbose(v => !v) }}
              className="sl-toast__verbose"
            >
              {showVerbose ? 'Hide' : 'Details'}
            </button>
          )}
        </span>
        <button
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
          className={`sl-toast__fill ${styles.bar}`}
          style={{ width: `${progress}%` }}
        />
      </span>
    </div>
  )
}

interface ToastContainerProps {
  toasts: Toast[]
  onDismiss: (id: string) => void
}

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  if (toasts.length === 0) return null
  return (
    <div className="sl-toast-container" role="region" aria-label="Notifications">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </div>
  )
}
