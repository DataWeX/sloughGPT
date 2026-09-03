'use client'

import Link from 'next/link'
import { Card, CardContent } from '@sloughgpt/strui'
import { formatBytes } from '@/lib/format-bytes'

interface UsageStatsProps {
  apiStatus: string
  convStats: { totalConversations: number; totalMessages: number; totalWords: number; activeDays: number; mostActiveHour: number | null } | null
  datasetStats: { totalDatasets: number; totalSize: number; totalSamples: number } | null
}

export function UsageStats({ apiStatus, convStats, datasetStats }: UsageStatsProps) {
  if (apiStatus !== 'online' || (!convStats && !datasetStats)) return null

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {convStats && convStats.totalConversations > 0 && (
        <Card>
          <CardContent className="py-3">
            <div className="flex items-center gap-2 mb-2">
              <p className="text-xs font-medium">Your stats</p>
              <p className="text-xs text-muted-foreground">Usage overview</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <p className="text-base font-semibold tabular-nums">{convStats.totalConversations}</p>
                <p className="text-xs text-muted-foreground">Conversations</p>
              </div>
              <div>
                <p className="text-base font-semibold tabular-nums">{convStats.totalMessages.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">Messages</p>
              </div>
              <div>
                <p className="text-base font-semibold tabular-nums">{convStats.totalWords.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">Words</p>
              </div>
              <div>
                <p className="text-base font-semibold tabular-nums">{convStats.activeDays}</p>
                <p className="text-xs text-muted-foreground">Active days</p>
              </div>
            </div>
            {convStats.mostActiveHour !== null && (
              <p className="text-xs text-muted-foreground mt-2">
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
              <p className="text-xs font-medium">Datasets</p>
              <Link href="/datasets" prefetch={false} className="text-xs text-primary hover:text-primary/80 ml-auto">View all →</Link>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <p className="text-base font-semibold tabular-nums">{datasetStats.totalDatasets}</p>
                <p className="text-xs text-muted-foreground">Datasets</p>
              </div>
              <div>
                <p className="text-base font-semibold tabular-nums">
                  {formatBytes(datasetStats.totalSize)}
                </p>
                <p className="text-xs text-muted-foreground">Total size</p>
              </div>
              <div>
                <p className="text-base font-semibold tabular-nums">{datasetStats.totalSamples.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">Samples</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
