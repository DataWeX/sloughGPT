'use client'

import { Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'
import { RefreshCw } from 'lucide-react'

interface TrainingStatus {
  completed: number
  running: number
  failed: number
}

interface DashboardTrainingOverviewProps {
  training: TrainingStatus
  totalMinutes: number
  totalJobs: number
  onRefresh?: () => void
}

export function DashboardTrainingOverview({
  training,
  totalMinutes,
  totalJobs,
  onRefresh,
}: DashboardTrainingOverviewProps) {
  const completedPct = totalJobs ? (training.completed / totalJobs) * 100 : 0
  const runningPct = totalJobs ? (training.running / totalJobs) * 100 : 0
  const failedPct = totalJobs ? (training.failed / totalJobs) * 100 : 0

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xs">Training Overview</CardTitle>
          {onRefresh && (
            <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onRefresh} title="Refresh">
              <RefreshCw className="h-3 w-3" />
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Completed</span>
            <span className="font-medium text-green-600">{training.completed}</span>
          </div>
          <div className="w-full bg-muted rounded-full h-1.5">
            <div className="bg-green-500 h-1.5 rounded-full" style={{ width: `${completedPct}%` }} />
          </div>

          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Running</span>
            <span className="font-medium text-blue-600">{training.running}</span>
          </div>
          <div className="w-full bg-muted rounded-full h-1.5">
            <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${runningPct}%` }} />
          </div>

          <div className="flex items-center justify-between text-[10px]">
            <span className="text-muted-foreground">Failed</span>
            <span className="font-medium text-red-600">{training.failed}</span>
          </div>
          <div className="w-full bg-muted rounded-full h-1.5">
            <div className="bg-red-500 h-1.5 rounded-full" style={{ width: `${failedPct}%` }} />
          </div>

          <div className="pt-2 border-t text-[10px] text-muted-foreground">
            Total training time: {totalMinutes} min
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
