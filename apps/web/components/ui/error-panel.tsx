'use client'

import { useState, useEffect, useRef } from 'react'
import { useErrorStore, type AppError, type ErrorSeverity } from '@/lib/error-store'
import { cn } from '@/lib/cn'

function ErrorIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function WarningIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
    </svg>
  )
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
    </svg>
  )
}

function ErrorItem({ error, onDismiss }: { error: AppError; onDismiss: (id: string) => void }) {
  const isError = error.severity === 'error'
  const isWarning = error.severity === 'warning'

  return (
    <div
      className={cn(
        "flex items-start gap-2 rounded-md border p-2.5 text-[11px]",
        isError && "border-red-200/60 bg-red-50/80 dark:border-red-900/50 dark:bg-red-950/40 text-red-800 dark:text-red-300",
        isWarning && "border-orange-200/60 bg-orange-50/80 dark:border-orange-900/50 dark:bg-orange-950/40 text-orange-800 dark:text-orange-300",
        error.severity === 'info' && "border-blue-200/60 bg-blue-50/80 dark:border-blue-900/50 dark:bg-blue-950/40 text-blue-800 dark:text-blue-300",
      )}
    >
      <span className="shrink-0 mt-0.5">
        {isError ? <ErrorIcon className="h-3.5 w-3.5" /> : <WarningIcon className="h-3.5 w-3.5" />}
      </span>
      <div className="flex-1 min-w-0">
        <p className="font-semibold leading-tight">{error.title}</p>
        {error.message && error.message !== error.title && (
          <p className="mt-0.5 opacity-80 leading-tight text-[10px] break-all max-h-16 overflow-y-auto">{error.message}</p>
        )}
        {error.source && (
          <p className="mt-0.5 opacity-60 text-[9px]">Source: {error.source}</p>
        )}
      </div>
      <button
        onClick={() => onDismiss(error.id)}
        className="shrink-0 p-0.5 rounded hover:opacity-70 transition-opacity"
        aria-label="Dismiss"
      >
        <CloseIcon className="h-3 w-3" />
      </button>
    </div>
  )
}

export function ErrorPanel() {
  const errors = useErrorStore(s => s.errors)
  const dismissError = useErrorStore(s => s.dismissError)
  const clearErrors = useErrorStore(s => s.clearErrors)
  const [open, setOpen] = useState(false)
  const prevCount = useRef(errors.length)

  useEffect(() => {
    if (errors.length > prevCount.current) {
      setOpen(true)
    }
    prevCount.current = errors.length
  }, [errors.length])

  const errorCount = errors.filter(e => e.severity === 'error').length
  const warningCount = errors.filter(e => e.severity === 'warning' || e.severity === 'info').length
  const total = errors.length

  if (total === 0) return null

  return (
    <div
      className={cn(
        "fixed bottom-4 right-4 z-[200]",
        "flex flex-col items-end gap-2"
      )}
    >
      <div
        className={cn(
          "overflow-hidden transition-all duration-300 ease-in-out",
          open ? "max-h-[60vh] opacity-100" : "max-h-0 opacity-0"
        )}
      >
        <div className="w-96 max-w-[calc(100vw-2rem)] space-y-1.5 p-2 bg-background/95 backdrop-blur-sm rounded-lg border border-border shadow-xl">
          <div className="flex items-center justify-between px-1 pb-1">
            <p className="text-[10px] font-semibold text-muted-foreground">
              {total} issue{total > 1 ? 's' : ''}
              {errorCount > 0 && ` · ${errorCount} error${errorCount > 1 ? 's' : ''}`}
            </p>
            {total > 1 && (
              <button
                onClick={clearErrors}
                className="text-[10px] text-muted-foreground hover:text-foreground underline"
              >
                Clear all
              </button>
            )}
          </div>
          {errors.map(error => (
            <ErrorItem key={error.id} error={error} onDismiss={dismissError} />
          ))}
        </div>
      </div>

      <button
        onClick={() => setOpen(o => !o)}
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-full text-[11px] font-semibold shadow-lg transition-all",
          errorCount > 0
            ? "bg-red-500 hover:bg-red-600 text-white"
            : "bg-orange-500 hover:bg-orange-600 text-white",
          "border-0"
        )}
      >
        {errorCount > 0 ? <ErrorIcon className="h-4 w-4" /> : <WarningIcon className="h-4 w-4" />}
        <span>{total} issue{total > 1 ? 's' : ''}</span>
        <ChevronIcon className={cn("h-3 w-3 transition-transform", open ? "rotate-180" : "")} />
      </button>
    </div>
  )
}