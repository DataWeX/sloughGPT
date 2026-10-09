'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, Button } from '@sloughgpt/strui'

interface ApiKeyRevealCardProps {
  revealedKey: string | null
  onDismiss?: () => void
}

export function ApiKeyRevealCard({ revealedKey, onDismiss }: ApiKeyRevealCardProps) {
  const [copied, setCopied] = useState(false)

  if (!revealedKey) return null

  const handleCopy = async () => {
    await navigator.clipboard.writeText(revealedKey)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card className="border-success/50 bg-success/5" data-testid="api-key-reveal">
      <CardHeader>
        <CardTitle className="text-success text-base">New API Key</CardTitle>
        <CardDescription>Copy this key now. It won&apos;t be shown again.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2">
          <code className="flex-1 p-2 bg-muted rounded text-xs break-all font-mono" data-testid="api-key-value">
            {revealedKey}
          </code>
          <Button size="sm" onClick={handleCopy}>
            {copied ? 'Copied!' : 'Copy'}
          </Button>
          {onDismiss && (
            <Button size="sm" variant="ghost" onClick={onDismiss}>Dismiss</Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
