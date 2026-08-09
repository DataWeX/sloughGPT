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
          <div className="rounded-md bg-muted/30 p-3 text-center">
            <div className="text-xs text-muted-foreground">Inferences</div>
            <div className="text-lg font-mono font-medium">{inferenceCount.toLocaleString()}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-3 text-center">
            <div className="text-xs text-muted-foreground">Requests</div>
            <div className="text-lg font-mono font-medium">{requestCount.toLocaleString()}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-3 text-center">
            <div className="text-xs text-muted-foreground">Active Model</div>
            <div className="text-sm font-medium truncate">{modelType ?? 'None'}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-3 text-center">
            <div className="text-xs text-muted-foreground">Avg Tokens</div>
            <div className="text-lg font-mono font-medium">
              {inferenceCount > 0 ? Math.round(requestCount / inferenceCount) : '—'}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
