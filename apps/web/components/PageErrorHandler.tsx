'use client'

import { useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { IconAlert, IconRefresh, IconCopy } from '@sloughgpt/strui'
import { addGlobalError } from '@/lib/error-store'
import { reportError } from '@/lib/error-reporter'
import { extractErrorMessage, formatStackTrace, getErrorType } from '@/lib/error-utils'
import { useEffect } from 'react'

interface PageErrorHandlerProps {
  error: Error & { digest?: string }
  reset: () => void
  title?: string
}

export function PageErrorHandler({ error, reset, title }: PageErrorHandlerProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [copied, setCopied] = useState(false)

  const errorMessage = extractErrorMessage(error, 'No error message')
  const errorType = getErrorType(error)
  const stackFrames = formatStackTrace(error.stack)
  const digest = (error as { digest?: string }).digest

  useEffect(() => {
    addGlobalError(error, 'PageErrorHandler')
    reportError(errorMessage, 'page-error-boundary', {
      stack: error.stack,
      metadata: { name: error.name, digest, url: window.location.href },
    })
  }, [error])

  const errorDetails = {
    title: title || 'Something went wrong',
    message: errorMessage,
    name: error.name,
    digest,
    stack: error.stack,
    url: window.location.href,
    timestamp: new Date().toISOString(),
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

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <Card className="shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10 shrink-0">
              <IconAlert className="h-5 w-5 text-destructive" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-base flex items-center gap-2">
                {title || 'Something went wrong'}
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
                  <span className="text-muted-foreground text-[10px] uppercase tracking-wider">Raw Stack</span>
                  <pre className="whitespace-pre-wrap break-all text-muted-foreground mt-0.5">{error.stack}</pre>
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
              Go home
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
