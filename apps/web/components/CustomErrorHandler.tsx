'use client'

import { useState, useEffect } from 'react'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { IconAlert, IconRefresh, IconCopy, IconChevronLeft } from '@sloughgpt/strui'
import { addGlobalError } from '@/lib/error-store'
import { reportError } from '@/lib/error-reporter'
import { extractErrorMessage, formatStackTrace, getErrorType } from '@/lib/error-utils'

interface CustomErrorHandlerProps {
  error: Error & { digest?: string }
  reset: () => void
}

export function CustomErrorHandler({ error, reset }: CustomErrorHandlerProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [copied, setCopied] = useState(false)

  const errorMessage = extractErrorMessage(error, 'No error message')
  const errorType = getErrorType(error)
  const stackFrames = formatStackTrace(error.stack)
  const digest = (error as { digest?: string }).digest

  useEffect(() => {
    addGlobalError(error, 'CustomErrorHandler')
    reportError(errorMessage, 'error-boundary', {
      stack: error.stack,
      metadata: { name: error.name, digest },
    })
  }, [error])

  const errorDetails = {
    message: errorMessage,
    name: error.name,
    digest,
    stack: error.stack,
    timestamp: new Date().toISOString(),
    url: typeof window !== 'undefined' ? window.location.href : 'unknown',
    userAgent: typeof window !== 'undefined' ? navigator.userAgent : 'unknown',
  }

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(errorDetails, null, 2))
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard not available
    }
  }

  const isNetworkError = errorMessage.toLowerCase().includes('fetch') ||
                         errorMessage.toLowerCase().includes('network') ||
                         errorMessage.toLowerCase().includes('econnrefused') ||
                         errorMessage.toLowerCase().includes('could not fetch')

  const isAuthError = errorMessage.includes('401') ||
                      errorMessage.toLowerCase().includes('unauthorized')

  const isNotFoundError = errorMessage.includes('404') ||
                          errorMessage.toLowerCase().includes('not found')

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="max-w-md w-full shadow-lg">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10 shrink-0">
              <IconAlert className="h-5 w-5 text-destructive" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-base flex items-center gap-2">
                {isNetworkError ? 'Connection Error' :
                 isAuthError ? 'Authentication Error' :
                 isNotFoundError ? 'Page Not Found' :
                 'Something went wrong'}
                {errorType && (
                  <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-destructive/10 text-destructive">
                    {errorType}
                  </span>
                )}
              </CardTitle>
              <p className="text-sm text-muted-foreground mt-0.5 break-words">
                {errorMessage}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors shrink-0"
            >
              {showDetails ? 'Hide' : 'Details'}
            </button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {showDetails && (
            <div className="rounded-md bg-muted p-3 text-xs font-mono space-y-3 max-h-64 overflow-y-auto">
              {digest && (
                <div>
                  <span className="text-muted-foreground text-[10px] uppercase tracking-wider">Digest</span>
                  <pre className="whitespace-pre-wrap break-all text-muted-foreground mt-0.5">{digest}</pre>
                </div>
              )}
              {stackFrames.length > 0 && (
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground text-[10px] uppercase tracking-wider">Stack Trace</span>
                    <button
                      type="button"
                      onClick={copyToClipboard}
                      className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <IconCopy className="h-3 w-3" />
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <pre className="whitespace-pre-wrap break-all text-muted-foreground mt-1">
                    {stackFrames.map((frame, i) => (
                      <div key={i}>{frame}</div>
                    ))}
                  </pre>
                </div>
              )}
              {error.stack && stackFrames.length === 0 && (
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground text-[10px] uppercase tracking-wider">Raw Stack</span>
                    <button
                      type="button"
                      onClick={copyToClipboard}
                      className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <IconCopy className="h-3 w-3" />
                      {copied ? 'Copied!' : 'Copy'}
                    </button>
                  </div>
                  <pre className="whitespace-pre-wrap break-all text-muted-foreground mt-1">{error.stack}</pre>
                </div>
              )}
            </div>
          )}

          <div className="flex items-center gap-2">
            <Button onClick={reset} className="flex-1" size="sm">
              <IconRefresh className="h-3.5 w-3.5 mr-1.5" />
              Try again
            </Button>
            <Button
              variant="outline"
              onClick={() => window.location.href = '/'}
              className="flex-1"
              size="sm"
            >
              <IconChevronLeft className="h-3.5 w-3.5 mr-1.5" />
              Go home
            </Button>
          </div>

          {isNetworkError && (
            <p className="text-xs text-muted-foreground text-center">
              Check your internet connection and try again. The service might be unreachable.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
