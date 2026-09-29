'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import { authFetch } from '@/lib/http-client'

const QUICK_ACTIONS = [
  {
    label: 'Restart Backend',
    description: 'Restart the FastAPI server',
    endpoint: '/system/restart',
    method: 'POST',
  },
  {
    label: 'Clear Cache',
    description: 'Clear model cache',
    endpoint: '/cache/clear',
    method: 'POST',
  },
  {
    label: 'Reset Metrics',
    description: 'Reset all metrics counters',
    endpoint: '/registry/stats/reset',
    method: 'POST',
  },
  {
    label: 'Reload Models',
    description: 'Reload model registry',
    endpoint: '/models/reload',
    method: 'POST',
  },
  { label: 'Health Check', description: 'Check system health', endpoint: '/health', method: 'GET' },
  {
    label: 'List Endpoints',
    description: 'List all API routes',
    endpoint: '/routes',
    method: 'GET',
  },
]

interface ActionResult {
  status: number
  body: string
  timeMs: number
}

const METHOD_COLORS: Record<string, string> = {
  GET: 'bg-success/10 text-success',
  POST: 'bg-warning/10 text-warning',
  PUT: 'bg-info/10 text-info',
  PATCH: 'bg-primary/10 text-primary',
  DELETE: 'bg-destructive/10 text-destructive',
}

export function DevQuickActionsCard() {
  const [results, setResults] = useState<Record<string, ActionResult | null>>({})
  const [running, setRunning] = useState<string | null>(null)

  const executeAction = async (action: (typeof QUICK_ACTIONS)[0]) => {
    setRunning(action.label)
    setResults((prev) => ({ ...prev, [action.label]: null }))
    const start = Date.now()

    try {
      const res = await authFetch(action.endpoint, { method: action.method, noAuth: true })
      const elapsed = Date.now() - start
      const text = await res.text()
      setResults((prev) => ({
        ...prev,
        [action.label]: { status: res.status, body: text, timeMs: elapsed },
      }))
    } catch (err) {
      setResults((prev) => ({
        ...prev,
        [action.label]: {
          status: 0,
          body: err instanceof Error ? err.message : 'Failed',
          timeMs: Date.now() - start,
        },
      }))
    } finally {
      setRunning(null)
    }
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-2.5 sm:grid-cols-2 lg:grid-cols-3">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            type="button"
            onClick={() => executeAction(action)}
            disabled={running !== null}
            className={cn(
              'group flex flex-col items-start p-4 rounded-xl border text-left transition-all duration-200',
              'hover:-translate-y-0.5 hover:shadow-lg hover:shadow-black/10',
              running === action.label
                ? 'border-primary/40 bg-primary/[0.04]'
                : results[action.label]
                  ? results[action.label]!.status >= 200 && results[action.label]!.status < 300
                    ? 'border-success/20 bg-success/[0.03]'
                    : 'border-destructive/20 bg-destructive/[0.03]'
                  : 'border-white/[0.06] bg-white/[0.02] hover:border-white/[0.12] hover:bg-white/[0.04]',
            )}
          >
            <div className="flex items-center gap-2 w-full mb-1">
              <span
                className={cn(
                  'text-[10px] font-mono font-medium px-1.5 py-0.5 rounded',
                  METHOD_COLORS[action.method] ?? 'bg-white/[0.06] text-muted-foreground',
                )}
              >
                {action.method}
              </span>
              <span className="text-sm font-medium text-foreground">{action.label}</span>
              {running === action.label && (
                <span className="ml-auto flex items-center gap-1.5 text-[10px] text-warning">
                  <span className="w-1.5 h-1.5 rounded-full bg-warning animate-pulse" />
                  running
                </span>
              )}
              {results[action.label] && running !== action.label && (
                <span
                  className={cn(
                    'ml-auto text-[10px] font-mono',
                    results[action.label]!.status >= 200 && results[action.label]!.status < 300
                      ? 'text-success'
                      : results[action.label]!.status >= 400
                        ? 'text-warning'
                        : 'text-destructive',
                  )}
                >
                  {results[action.label]!.status || 'ERR'}
                </span>
              )}
            </div>
            <span className="text-[11px] text-muted-foreground">{action.description}</span>
            <span className="text-[10px] text-muted-foreground font-mono mt-1.5">
              {action.endpoint}
            </span>
          </button>
        ))}
      </div>

      {Object.entries(results).map(
        ([label, result]) =>
          result && (
            <div
              key={label}
              className="rounded-xl border border-white/[0.06] bg-background overflow-hidden"
            >
              <div className="flex items-center justify-between px-4 py-2.5 bg-muted border-b border-white/[0.06]">
                <span className="text-[11px] font-medium text-foreground">{label}</span>
                <div className="flex items-center gap-2">
                  <span
                    className={cn(
                      'text-[10px] font-mono',
                      result.status >= 200 && result.status < 300
                        ? 'text-success'
                        : result.status >= 400
                          ? 'text-warning'
                          : 'text-destructive',
                    )}
                  >
                    {result.status || 'ERR'}
                  </span>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    {result.timeMs}ms
                  </span>
                </div>
              </div>
              <pre className="px-4 py-3 text-[11px] font-mono text-foreground overflow-auto max-h-48 whitespace-pre-wrap leading-relaxed">
                {result.body || 'No response'}
              </pre>
            </div>
          ),
      )}
    </div>
  )
}
