'use client'

import Link from 'next/link'
import { Card, CardContent } from '@sloughgpt/strui'
import { formatBytes } from '@/lib/format-bytes'

interface UsageStatsProps {
  apiStatus: string
  loading: boolean
  convStats: { totalConversations: number; totalMessages: number; totalWords: number; activeDays: number; mostActiveHour: number | null } | null
  datasetStats: { totalDatasets: number; totalSize: number; totalSamples: number } | null
}

export function UsageStats({ apiStatus, loading, convStats, datasetStats }: UsageStatsProps) {
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <Card>
          <CardContent className="py-3">
            <div className="h-4 w-24 animate-pulse rounded bg-muted mb-2" />
            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="space-y-1">
                  <div className="h-5 w-12 animate-pulse rounded bg-muted" />
                  <div className="h-3 w-16 animate-pulse rounded bg-muted" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-3">
            <div className="h-4 w-20 animate-pulse rounded bg-muted mb-2" />
            <div className="grid grid-cols-2 gap-3">
              {Array.from({ length: 2 }).map((_, i) => (
                <div key={i} className="space-y-1">
                  <div className="h-5 w-12 animate-pulse rounded bg-muted" />
                  <div className="h-3 w-16 animate-pulse rounded bg-muted" />
                </div>
              ))}
            </div>
            <div className="mt-3 pt-3 border-t border-border/40">
              <div className="h-5 w-12 animate-pulse rounded bg-muted" />
              <div className="h-3 w-16 animate-pulse rounded bg-muted" />
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (apiStatus !== 'online' || (!convStats && !datasetStats)) return null

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {convStats && convStats.totalConversations > 0 && (
        <Card>
          <CardContent className="py-3">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Your stats</p>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Conversations</p>
                <p className="text-sm font-semibold tabular-nums">{convStats.totalConversations}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Messages</p>
                <p className="text-sm font-semibold tabular-nums">{convStats.totalMessages.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Words</p>
                <p className="text-sm font-semibold tabular-nums">{convStats.totalWords.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Active days</p>
                <p className="text-sm font-semibold tabular-nums">{convStats.activeDays}</p>
              </div>
            </div>
            {convStats.mostActiveHour !== null && (
              <p className="text-[10px] text-muted-foreground/60 mt-2">
                Most active at {convStats.mostActiveHour}:00
              </p>
            )}
          </CardContent>
        </Card>
      )}
      {datasetStats && datasetStats.totalDatasets > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-2 mb-2">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Datasets</p>
              <Link href="/datasets" prefetch={false} className="text-[10px] text-primary hover:text-primary/80 ml-auto">View all →</Link>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Datasets</p>
                <p className="text-sm font-semibold tabular-nums">{datasetStats.totalDatasets}</p>
              </div>
              <div>
                <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Total size</p>
                <p className="text-sm font-semibold tabular-nums">
                  {formatBytes(datasetStats.totalSize)}
                </p>
              </div>
            </div>
            <div className="mt-2 pt-2 border-t border-border/30">
              <p className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Samples</p>
              <p className="text-sm font-semibold tabular-nums">{datasetStats.totalSamples.toLocaleString()}</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
