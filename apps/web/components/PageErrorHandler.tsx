'use client'

import { useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { IconAlert, IconRefresh } from '@sloughgpt/strui'
import { addGlobalError } from '@/lib/error-store'
import { reportError } from '@/lib/error-reporter'
import { useEffect } from 'react'

interface PageErrorHandlerProps {
  error: Error & { digest?: string }
  reset: () => void
  title?: string
}

export function PageErrorHandler({ error, reset, title }: PageErrorHandlerProps) {
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    addGlobalError(error, 'PageErrorHandler')
    reportError(error.message, 'page-error-boundary', {
      stack: error.stack,
      metadata: { name: error.name, digest: error.digest, url: window.location.href },
    })
  }, [error])

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <Card className="shadow-sm">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-destructive/10 shrink-0">
              <IconAlert className="h-5 w-5 text-destructive" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-base">
                {title || 'Something went wrong'}
              </CardTitle>
              <p className="text-xs text-muted-foreground mt-0.5 truncate">
                {error.message.slice(0, 80)}
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowDetails(!showDetails)}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showDetails ? 'Hide' : 'Details'}
            </button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {showDetails && (
            <div className="rounded-md bg-muted p-3 text-xs font-mono max-h-48 overflow-y-auto">
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
              Go home
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
