'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, cn } from '@sloughgpt/strui'
import { authFetch } from '@/lib/http-client'

const QUICK_ACTIONS = [
  { label: 'Restart Backend', description: 'Restart the FastAPI server', endpoint: '/system/restart', method: 'POST' },
  { label: 'Clear Cache', description: 'Clear model cache', endpoint: '/cache/clear', method: 'POST' },
  { label: 'Reset Metrics', description: 'Reset all metrics counters', endpoint: '/registry/stats/reset', method: 'POST' },
  { label: 'Reload Models', description: 'Reload model registry', endpoint: '/models/reload', method: 'POST' },
  { label: 'Health Check', description: 'Check system health', endpoint: '/health', method: 'GET' },
  { label: 'List Endpoints', description: 'List all API routes', endpoint: '/routes', method: 'GET' },
]

interface ActionResult {
  status: number
  body: string
  timeMs: number
}

export function DevQuickActionsCard() {
  const [results, setResults] = useState<Record<string, ActionResult | null>>({})
  const [running, setRunning] = useState<string | null>(null)

  const executeAction = async (action: typeof QUICK_ACTIONS[0]) => {
    setRunning(action.label)
    setResults(prev => ({ ...prev, [action.label]: null }))
    const start = Date.now()

    try {
      const res = await authFetch(action.endpoint, { method: action.method, noAuth: true })
      const elapsed = Date.now() - start
      const text = await res.text()
      setResults(prev => ({ ...prev, [action.label]: { status: res.status, body: text, timeMs: elapsed } }))
    } catch (err) {
      setResults(prev => ({ ...prev, [action.label]: { status: 0, body: err instanceof Error ? err.message : 'Failed', timeMs: Date.now() - start } }))
    } finally {
      setRunning(null)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Quick Actions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {QUICK_ACTIONS.map(action => (
            <button
              key={action.label}
              type="button"
              onClick={() => executeAction(action)}
              disabled={running !== null}
              className={cn(
                'flex flex-col items-start p-3 rounded-lg border text-left transition-all hover:-translate-y-0.5 hover:shadow-sm',
                running === action.label ? 'border-primary/50 bg-primary/5' : 'border-border/60 hover:border-primary/30',
                results[action.label]?.status === 200 ? 'border-green-500/30 bg-green-500/5' :
                results[action.label]?.status === 404 ? 'border-yellow-500/30 bg-yellow-500/5' :
                results[action.label]?.status === 0 ? 'border-red-500/30 bg-red-500/5' : ''
              )}
            >
              <div className="flex items-center gap-2 w-full">
                <span className="text-sm font-medium">{action.label}</span>
                {running === action.label && (
                  <span className="ml-auto text-[10px] text-primary animate-pulse">Running...</span>
                )}
                {results[action.label] && running !== action.label && (
                  <span className={cn(
                    'ml-auto text-[10px] font-mono',
                    results[action.label]!.status >= 200 && results[action.label]!.status < 300 ? 'text-green-500' :
                    results[action.label]!.status >= 400 ? 'text-yellow-500' : 'text-red-500'
                  )}>
                    {results[action.label]!.status || 'ERR'} · {results[action.label]!.timeMs}ms
                  </span>
                )}
              </div>
              <span className="text-xs text-muted-foreground">{action.description}</span>
              <span className="text-[10px] text-muted-foreground/60 font-mono mt-1">{action.method} {action.endpoint}</span>
            </button>
          ))}
        </div>

        {Object.entries(results).map(([label, result]) => result && (
          <div key={label} className="mt-3">
            <p className="text-sm font-medium mb-1">{label}</p>
            <pre className="text-xs font-mono bg-muted/30 rounded p-3 overflow-auto max-h-48 whitespace-pre-wrap">
              {result.body || 'No response'}
            </pre>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
