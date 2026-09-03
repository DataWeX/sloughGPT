'use client'

import Link from 'next/link'
import { Card, CardContent } from '@sloughgpt/strui'
import { formatUptime } from '@/lib/chat-utils'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface SystemHealthProps {
  apiStatus: string
  loading: boolean
  liveHealth: LiveHealthSnapshot | null
}

export function SystemHealth({ apiStatus, loading, liveHealth }: SystemHealthProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Card>
          <CardContent className="py-3">
            <div className="h-4 w-16 animate-pulse rounded bg-muted mb-2" />
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="space-y-1">
                  <div className="h-5 w-10 animate-pulse rounded bg-muted" />
                  <div className="h-3 w-12 animate-pulse rounded bg-muted" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <div className="rounded-lg border border-border/60 p-3 sm:p-4 space-y-2">
          <div className="h-3 w-24 animate-pulse rounded bg-muted" />
          <div className="h-3 w-full animate-pulse rounded bg-muted" />
          <div className="h-3 w-3/4 animate-pulse rounded bg-muted" />
        </div>
      </div>
    )
  }

  if (apiStatus !== 'online' || !liveHealth) return null

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-2 mb-2">
            <p className="text-xs font-medium">System</p>
            <Link href="/monitoring" prefetch={false} className="text-xs text-primary hover:text-primary/80 ml-auto">Details →</Link>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div>
              <p className="text-sm font-semibold tabular-nums">
                {liveHealth.cpu_percent !== null ? `${Math.round(liveHealth.cpu_percent)}%` : '—'}
              </p>
              <p className="text-xs text-muted-foreground">CPU</p>
            </div>
            <div>
              <p className="text-sm font-semibold tabular-nums">
                {liveHealth.memory_percent !== null ? `${Math.round(liveHealth.memory_percent)}%` : '—'}
              </p>
              <p className="text-xs text-muted-foreground">Memory</p>
            </div>
            <div>
              <p className="text-sm font-semibold tabular-nums">{(liveHealth.request_count ?? 0).toLocaleString()}</p>
              <p className="text-xs text-muted-foreground">Requests</p>
            </div>
            <div>
              <p className="text-sm font-semibold tabular-nums">
                {liveHealth.uptime_seconds > 0 ? formatUptime(liveHealth.uptime_seconds) : '—'}
              </p>
              <p className="text-xs text-muted-foreground">Uptime</p>
            </div>
          </div>
        </CardContent>
      </Card>
      <div className="rounded-lg border border-border/60 p-3 sm:p-4 flex flex-col justify-between">
        <div>
          <div className="text-xs font-medium text-muted-foreground mb-2">How it works</div>
          <div className="space-y-2">
            <p className="text-[11px] text-muted-foreground/70">
              Mix and match AI models with personalities. Each one has its own voice and style.
            </p>
            <p className="text-[11px] text-muted-foreground/70">
              Import text, files, or conversations. The AI learns from your data and gets better over time.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
