'use client'

import { Card, CardContent, Skeleton } from '@sloughgpt/strui'
import type { MemoryStats } from '@/lib/memory-controller'

interface MemoryStatsGridProps {
  stats: MemoryStats | null
  loading: boolean
}

export function MemoryStatsGrid({ stats, loading }: MemoryStatsGridProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
      <div className="rounded-lg border border-border/60 p-3">
        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Facts</p>
        {loading ? <Skeleton className="h-6 w-12 mt-1" /> : <p className="text-xl font-semibold mt-1">{stats?.total_facts ?? 0}</p>}
      </div>
      <div className="rounded-lg border border-border/60 p-3">
        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Topics</p>
        {loading ? <Skeleton className="h-6 w-12 mt-1" /> : <p className="text-xl font-semibold mt-1">{stats?.topics ?? 0}</p>}
      </div>
      <div className="rounded-lg border border-border/60 p-3">
        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Visited URLs</p>
        {loading ? <Skeleton className="h-6 w-12 mt-1" /> : <p className="text-xl font-semibold mt-1">{stats?.visited_urls ?? 0}</p>}
      </div>
      <div className="rounded-lg border border-border/60 p-3">
        <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Searchable</p>
        {loading ? <Skeleton className="h-6 w-12 mt-1" /> : (
          <p className="text-xl font-semibold mt-1">{stats?.enabled ? 'Yes' : 'No'}</p>
        )}
      </div>
    </div>
  )
}
