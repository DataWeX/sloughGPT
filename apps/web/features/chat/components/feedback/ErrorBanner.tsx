'use client'

import { useCallback } from 'react'
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
    correlationId?: string
    requestId?: string
    backendError?: string
    statusCode?: number
    httpMethod?: string
    httpPath?: string
    durationMs?: number
  }
  onRetry: () => void
  onDismiss: () => void
}

export function ErrorBanner({ error, onRetry, onDismiss }: ErrorBannerProps) {
  const info = ERROR_MESSAGES[error.type]

  const handleCopyDiagnostics = useCallback(() => {
    const lines = [
      `Error: ${info.title}`,
      `Message: ${error.message}`,
      `Type: ${error.type}`,
      `Status: ${error.statusCode ?? 'N/A'}`,
      `Request: ${error.httpMethod && error.httpPath ? `${error.httpMethod} ${error.httpPath}` : 'N/A'}`,
      `Duration: ${error.durationMs != null ? `${error.durationMs}ms` : 'N/A'}`,
      `Correlation ID: ${error.correlationId ?? error.requestId ?? 'N/A'}`,
      `Backend: ${error.backendError ?? 'N/A'}`,
      `Time: ${new Date().toISOString()}`,
    ]
    navigator.clipboard.writeText(lines.join('\n')).catch(() => {})
  }, [error, info])

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
          {error.httpMethod && error.httpPath && (
            <p className="mt-0.5 text-[10px] text-destructive/50 font-mono">
              {error.httpMethod} {error.httpPath}
              {error.statusCode != null && (
                <span className="ml-1">→ {error.statusCode}</span>
              )}
              {error.durationMs != null && (
                <span className="ml-1">({error.durationMs}ms)</span>
              )}
            </p>
          )}
          {(error.correlationId || error.requestId) && (
            <p className="mt-0.5 text-[10px] text-destructive/40 font-mono">
              ID: {error.correlationId || error.requestId}
            </p>
          )}
          <p className="mt-0.5 text-[10px] text-destructive/50">
            {info.suggestion}
          </p>
        </div>
        <div className="flex shrink-0 gap-1.5 sm:gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopyDiagnostics}
            className="h-7 text-xs sm:h-8 text-muted-foreground hover:text-foreground"
            aria-label="Copy error diagnostics"
            title="Copy diagnostics to clipboard"
          >
            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15.666 3.888A2.25 2.25 0 0013.5 2.25h-3c-1.03 0-1.9.693-2.166 1.638m7.332 0c.055.194.084.4.084.612v0a.75.75 0 01-.75.75H9.75a.75.75 0 01-.75-.75v0c0-.212.03-.418.084-.612m7.332 0c.646.049 1.288.11 1.927.184 1.1.128 1.907 1.077 1.907 2.185V19.5a2.25 2.25 0 01-2.25 2.25H6.75A2.25 2.25 0 014.5 19.5V6.257c0-1.108.806-2.057 1.907-2.185a48.208 48.208 0 011.927-.184" />
            </svg>
          </Button>
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

export function getErrorInfo(
  status: number,
  message?: string,
  opts?: { correlationId?: string; requestId?: string; backendError?: string; httpMethod?: string; httpPath?: string; durationMs?: number },
): { type: ErrorType; message: string; canRetry: boolean; correlationId?: string; requestId?: string; backendError?: string; statusCode?: number; httpMethod?: string; httpPath?: string; durationMs?: number } {
  const base = { correlationId: opts?.correlationId, requestId: opts?.requestId, backendError: opts?.backendError, statusCode: status, httpMethod: opts?.httpMethod, httpPath: opts?.httpPath, durationMs: opts?.durationMs }

  if (status === 0) {
    return { type: 'network', message: 'Could not connect to the service.', canRetry: true, ...base }
  }
  if (status === 400) {
    return { type: 'server', message: message || 'Invalid request.', canRetry: false, ...base }
  }
  if (status === 401) {
    return { type: 'server', message: message || 'Authentication required.', canRetry: false, ...base }
  }
  if (status === 403) {
    return { type: 'server', message: message || 'Access denied.', canRetry: false, ...base }
  }
  if (status === 404) {
    return { type: 'server', message: 'Chat endpoint not found. Please try again.', canRetry: true, ...base }
  }
  if (status === 408) {
    return { type: 'timeout', message: message || 'Request timed out.', canRetry: true, ...base }
  }
  if (status === 422) {
    return { type: 'server', message: message || 'Invalid request format.', canRetry: false, ...base }
  }
  if (status === 429) {
    return { type: 'server', message: message || 'Too many requests. Please slow down.', canRetry: true, ...base }
  }
  if (status === 500) {
    return { type: 'server', message: message || 'Internal server error.', canRetry: true, ...base }
  }
  if (status === 502) {
    return { type: 'server', message: message || 'Service temporarily unavailable.', canRetry: true, ...base }
  }
  if (status === 503) {
    return { type: 'model', message: message || 'Model not available.', canRetry: true, ...base }
  }
  if (status === 504) {
    return { type: 'timeout', message: message || 'Generation timed out.', canRetry: true, ...base }
  }
  return { type: 'unknown', message: message || `HTTP ${status}`, canRetry: true, ...base }
}
