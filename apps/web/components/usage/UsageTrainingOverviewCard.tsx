'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

export interface TrainingStatus {
  completed: number
  running: number
  queued: number
  failed: number
}

export interface UsageTrainingOverviewCardProps {
  status: TrainingStatus
  totalMinutes: number
}

export function UsageTrainingOverviewCard({ status, totalMinutes }: UsageTrainingOverviewCardProps) {
  const stats = [
    { label: 'Completed', value: status.completed, color: 'text-green-500' },
    { label: 'Running', value: status.running, color: 'text-blue-500' },
    { label: 'Queued', value: status.queued, color: 'text-yellow-500' },
    { label: 'Failed', value: status.failed, color: 'text-red-500' },
  ]

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-xs">Training Jobs</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {stats.map(s => (
            <div key={s.label} className="text-center p-2 rounded bg-muted/30">
              <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
              <div className="text-[10px] text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-[10px] text-muted-foreground text-center">
          Total training time: {totalMinutes} minutes
        </div>
      </CardContent>
    </Card>
  )
}
