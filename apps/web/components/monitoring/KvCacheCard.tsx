'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import type { KvSessionsInfo } from '@/lib/system-controller'

interface KvCacheCardProps {
  kvSessions: KvSessionsInfo
}

export const KvCacheCard = memo(function KvCacheCard({ kvSessions }: KvCacheCardProps) {
  if (!kvSessions.enabled) return null

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">KV cache sessions</span>
        <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-success/10 text-success text-[10px] font-medium">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-success" />
          cross-turn reuse
        </span>
      </div>
      <CardContent className="p-0">
        <KpiGrid columns={4}>
          <StatCard
            label="Active"
            value={kvSessions.active_sessions ?? 0} numeric
            icon={<span className={`inline-block w-2 h-2 rounded-full ${(kvSessions.active_sessions ?? 0) > 0 ? 'bg-success' : 'bg-muted-foreground/50'}`} />}
          />
          <StatCard label="Cached tokens" value={kvSessions.cached_tokens ?? 0} numeric />
          <StatCard label="TTL" value={(kvSessions.ttl_seconds ?? 0) / 60 + "m"} numeric />
          <StatCard
            label="Oldest"
            value={kvSessions.oldest_session_age != null ? `${kvSessions.oldest_session_age.toFixed(0)}s` : <Skeleton className="h-5 w-10" />} numeric
          />
        </KpiGrid>
        {kvSessions.max_sessions != null && (
          <p className="text-xs text-muted-foreground px-3 pb-3">
            LRU cap: {kvSessions.max_sessions} simultaneous sessions
          </p>
        )}
      </CardContent>
    </Card>
  )
})
