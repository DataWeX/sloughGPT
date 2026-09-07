'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'

interface ModelUsageCardProps {
  inferenceCount: number
  requestCount: number
  modelType: string | null
  isOnline: boolean
}

export default function ModelUsageCard({ inferenceCount, requestCount, modelType, isOnline }: ModelUsageCardProps) {
  if (!isOnline) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Usage Statistics</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-lg bg-muted/20 px-3 py-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Inferences</div>
            <div className="text-sm font-semibold tabular-nums mt-0.5">{inferenceCount.toLocaleString()}</div>
          </div>
          <div className="rounded-lg bg-muted/20 px-3 py-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Requests</div>
            <div className="text-sm font-semibold tabular-nums mt-0.5">{requestCount.toLocaleString()}</div>
          </div>
          <div className="rounded-lg bg-muted/20 px-3 py-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Active Model</div>
            <div className="text-sm font-medium truncate mt-0.5">{modelType ?? 'None'}</div>
          </div>
          <div className="rounded-lg bg-muted/20 px-3 py-2">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Avg Tokens</div>
            <div className="text-sm font-semibold tabular-nums mt-0.5">
              {inferenceCount > 0 ? Math.round(requestCount / inferenceCount) : '—'}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
