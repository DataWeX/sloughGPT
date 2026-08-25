'use client'

import { Button } from '@sloughgpt/strui'

type ErrorType = 'network' | 'server' | 'model' | 'timeout' | 'unknown'

const ERROR_MESSAGES: Record<ErrorType, { title: string; suggestion: string }> = {
  network: {
    title: 'Connection failed',
    suggestion: 'Check if the service is running.',
  },
  server: {
    title: 'Service error',
    suggestion: 'The request returned an error. Try again or restart the service.',
  },
  model: {
    title: 'Model not loaded',
    suggestion: 'No model is currently loaded. Load a model first.',
  },
  timeout: {
    title: 'Request timed out',
    suggestion: 'The model took too long. Try a shorter response.',
  },
  unknown: {
    title: 'Something went wrong',
    suggestion: 'An unexpected error occurred. Please try again.',
  },
}

interface ErrorBannerProps {
  error: {
    type: ErrorType
    message: string
    canRetry: boolean
  }
  onRetry: () => void
  onDismiss: () => void
}

export function ErrorBanner({ error, onRetry, onDismiss }: ErrorBannerProps) {
  const info = ERROR_MESSAGES[error.type]

  return (
    <section
      className="shrink-0 border-b border-destructive/20 bg-destructive/5 px-3 py-2.5 sm:px-4 sm:py-3"
      role="alert"
      aria-live="assertive"
    >
      <div className="mx-auto flex max-w-2xl items-start gap-2 sm:gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-destructive sm:text-sm">
            {info.title}
          </p>
          <p className="mt-0.5 text-xs text-destructive/70">
            {error.message}
          </p>
          <p className="mt-0.5 text-[10px] text-destructive/50">
            {info.suggestion}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5 sm:gap-2">
          {error.canRetry && (
            <Button
              variant="outline"
              size="sm"
              onClick={onRetry}
              className="h-7 text-xs sm:h-8 border-destructive/30 text-destructive hover:bg-destructive/10"
              aria-label={`Retry ${info.title.toLowerCase()}`}
            >
              Retry
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            onClick={onDismiss}
            className="h-7 text-xs sm:h-8 text-muted-foreground"
            aria-label="Dismiss error message"
          >
            Dismiss
          </Button>
        </div>
      </div>
    </section>
  )
}

export function getErrorInfo(status: number, message?: string): { type: ErrorType; message: string; canRetry: boolean } {
  if (status === 0) {
    return { type: 'network', message: 'Could not connect to the service.', canRetry: true }
  }
  if (status === 400) {
    return { type: 'server', message: message || 'Invalid request.', canRetry: false }
  }
  if (status === 404) {
    return { type: 'server', message: 'Chat endpoint not found. Please try again.', canRetry: true }
  }
  if (status === 408) {
    return { type: 'timeout', message: message || 'Request timed out.', canRetry: true }
  }
  if (status === 422) {
    return { type: 'server', message: message || 'Invalid request format.', canRetry: false }
  }
  if (status === 429) {
    return { type: 'server', message: message || 'Too many requests. Please slow down.', canRetry: true }
  }
  if (status === 500) {
    return { type: 'server', message: message || 'Internal server error.', canRetry: true }
  }
  if (status === 502) {
    return { type: 'server', message: message || 'Service temporarily unavailable.', canRetry: true }
  }
  if (status === 503) {
    return { type: 'model', message: message || 'Model not available.', canRetry: true }
  }
  if (status === 504) {
    return { type: 'timeout', message: message || 'Generation timed out.', canRetry: true }
  }
  return { type: 'unknown', message: message || `HTTP ${status}`, canRetry: true }
}
