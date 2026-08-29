'use client'

import { memo } from 'react'
import { StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'

interface Props {
  checkpointCount: number
  completedCount: number
  trainingRunning: boolean
  loss: number | null
  loading: boolean
}

export const AutoTrainKpiGrid = memo(function AutoTrainKpiGrid({ checkpointCount, completedCount, trainingRunning, loss, loading }: Props) {
  const v = (n: number | string) => loading ? <Skeleton className="h-5 w-8 inline-block" /> : n
  return (
    <KpiGrid columns={4}>
      <StatCard label="Checkpoints" value={v(checkpointCount)} />
      <StatCard label="Trained" value={v(completedCount)} />
      <StatCard label="Status" value={v(trainingRunning ? 'Training' : 'Idle')} />
      <StatCard label="Current loss" value={v(loss != null ? loss.toFixed(4) : '--')} />
    </KpiGrid>
  )
})
