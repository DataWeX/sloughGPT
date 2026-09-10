'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export interface TrainingJobsCardProps {
  completed: number
  running: number
  queued: number
  failed: number
  totalMinutes: number
}

export function TrainingJobsCard({ completed, running, queued, failed, totalMinutes }: TrainingJobsCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Training Jobs</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="text-center p-2 rounded bg-muted/30">
            <div className="text-lg font-bold text-green-500">{completed}</div>
            <div className="text-[10px] text-muted-foreground">Completed</div>
          </div>
          <div className="text-center p-2 rounded bg-muted/30">
            <div className="text-lg font-bold text-blue-500">{running}</div>
            <div className="text-[10px] text-muted-foreground">Running</div>
          </div>
          <div className="text-center p-2 rounded bg-muted/30">
            <div className="text-lg font-bold text-yellow-500">{queued}</div>
            <div className="text-[10px] text-muted-foreground">Queued</div>
          </div>
          <div className="text-center p-2 rounded bg-muted/30">
            <div className="text-lg font-bold text-red-500">{failed}</div>
            <div className="text-[10px] text-muted-foreground">Failed</div>
          </div>
        </div>
        <div className="mt-3 text-[10px] text-muted-foreground text-center">
          Total training time: {totalMinutes} minutes
        </div>
      </CardContent>
    </Card>
  )
}
