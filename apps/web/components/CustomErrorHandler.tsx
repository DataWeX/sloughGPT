'use client'

import { useState, useEffect } from 'react'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { IconAlert, IconRefresh, IconCopy, IconX } from '@sloughgpt/strui'
import { addGlobalError } from '@/lib/error-store'
import { reportError } from '@/lib/error-reporter'

interface CustomErrorHandlerProps {
  error: Error & { digest?: string }
  reset: () => void
}

export function CustomErrorHandler({ error, reset }: CustomErrorHandlerProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    addGlobalError(error, 'CustomErrorHandler')
    reportError(error.message, 'error-boundary', {
      stack: error.stack,
      metadata: { name: error.name, digest: error.digest },
    })
  }, [error])

  const errorDetails = {
    message: error.message,
    name: error.name,
    stack: error.stack,
    digest: error.digest,
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

  const isNetworkError = error.message.includes('fetch') ||
                         error.message.includes('network') ||
                         error.message.includes('ECONNREFUSED') ||
                         error.message.includes('Failed to fetch')

  const isAuthError = error.message.includes('401') ||
                      error.message.includes('Unauthorized')

  const isNotFoundError = error.message.includes('404') ||
                          error.message.includes('Not Found')

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="max-w-md w-full shadow-lg">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10 shrink-0">
              <IconAlert className="h-5 w-5 text-destructive" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-base font-semibold">
                {isNetworkError ? 'Connection Error' :
                 isAuthError ? 'Authentication Error' :
                 isNotFoundError ? 'Page Not Found' :
                 'Something went wrong'}
              </CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5 truncate">
                {error.message.slice(0, 80)}
              </p>
            </div>
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showDetails ? 'Hide' : 'Details'}
            </button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {showDetails && (
            <div className="rounded-md bg-muted p-3 text-xs font-mono space-y-2 max-h-48 overflow-y-auto">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Error Details</span>
                <button
                  onClick={copyToClipboard}
                  className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
                >
                  <IconCopy className="h-3 w-3" />
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <pre className="whitespace-pre-wrap break-all text-muted-foreground">
                {error.stack || error.message}
              </pre>
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
              <svg className="h-3.5 w-3.5 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
              Go home
            </Button>
          </div>

          {isNetworkError && (
            <p className="text-xs text-muted-foreground text-center">
              Check your internet connection and try again. The server might be unreachable.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
