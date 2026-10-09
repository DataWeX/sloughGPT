'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'
import { PUBLIC_API_URL } from '@/lib/config'
import { modelController } from '@/lib/model-controller'
import { extractErrorMessage } from '@/lib/error-utils'

interface SettingsConnectionCardProps {
  apiUrl: string
  hfToken: string
  onApiUrlChange: (url: string) => void
  onHfTokenChange: (token: string) => void
}

export function SettingsConnectionCard({
  apiUrl,
  hfToken,
  onApiUrlChange,
  onHfTokenChange,
}: SettingsConnectionCardProps) {
  const [connectionTest, setConnectionTest] = useState<{ status: 'idle' | 'testing' | 'ok' | 'error'; latency?: number; error?: string }>({ status: 'idle' })

  const handleTestConnection = async () => {
    setConnectionTest({ status: 'testing' })
    const start = Date.now()
    try {
      const h = await modelController.getHealth()
      const latency = Date.now() - start
      if (h && (h.status === 'healthy' || h.model_loaded !== undefined)) {
        setConnectionTest({ status: 'ok', latency })
      } else {
        setConnectionTest({ status: 'error', error: 'Unexpected response' })
      }
    } catch (e: unknown) {
      setConnectionTest({ status: 'error', error: extractErrorMessage(e, 'Could not connect') })
    }
  }

  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-base">Connection</CardTitle>
          <CardDescription>Service connection and authentication</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="settings-api-url" className="text-sm font-medium">API URL</label>
          <Input
            id="settings-api-url"
            value={apiUrl}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => onApiUrlChange(e.target.value)}
            placeholder={PUBLIC_API_URL}
            className="font-mono text-xs"
            aria-label="Service URL"
          />
          <p className="text-[11px] text-muted-foreground">Service address. Changes take effect on next request.</p>
        </div>
        <div className="space-y-2">
          <label htmlFor="settings-hf-token" className="text-sm font-medium">HuggingFace Token</label>
          <Input
            id="settings-hf-token"
            type="password"
            value={hfToken}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => onHfTokenChange(e.target.value)}
            placeholder="hf_..."
            className="font-mono text-xs"
            aria-label="HuggingFace API token"
          />
          <p className="text-xs text-muted-foreground">Required for loading private HuggingFace models. Stored locally in your browser.</p>
        </div>
        <div className="flex items-center gap-2 pt-2">
          <Button size="sm" variant="outline" className="h-9 text-xs" onClick={handleTestConnection} disabled={connectionTest.status === 'testing'}>
            {connectionTest.status === 'testing' ? 'Testing...' : 'Test connection'}
          </Button>
          {connectionTest.status === 'ok' && (
            <span className="text-xs text-success font-medium">Connected ({connectionTest.latency}ms)</span>
          )}
          {connectionTest.status === 'error' && (
            <span className="text-xs text-destructive font-medium">{connectionTest.error}</span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
