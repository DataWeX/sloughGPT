'use client'

import { Card, CardHeader, CardTitle, CardContent, StatCard, KpiGrid, cn } from '@sloughgpt/strui'
import dynamicNext from '@/vite/next-compat/dynamic'
import type { LossPoint, RewardPoint } from '@/components/training/LossChart'
import type { TrainingJob } from '@/lib/training-controller'

const LossChart = dynamicNext(
  () => import('@/components/training/LossChart').then((m) => m.LossChart),
  { ssr: false },
)

interface JobLossCardProps {
  job: TrainingJob
}

export function JobLossCard({ job }: JobLossCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Loss &amp; Reward</CardTitle>
          {job.loss_history &&
            job.loss_history.length >= 3 &&
            (() => {
              const recent = job.loss_history.slice(-5)
              const firstHalf = recent.slice(0, Math.floor(recent.length / 2))
              const secondHalf = recent.slice(Math.floor(recent.length / 2))
              const avgFirst = firstHalf.reduce((s, p) => s + p.value, 0) / firstHalf.length
              const avgSecond = secondHalf.reduce((s, p) => s + p.value, 0) / secondHalf.length
              const improving = avgSecond < avgFirst
              const pctChange =
                avgFirst > 0 ? (((avgSecond - avgFirst) / avgFirst) * 100).toFixed(1) : '0'
              return (
                <span
                  className={cn(
                    'text-xs font-medium px-2 py-0.5 rounded',
                    improving ? 'bg-success/10 text-success' : 'bg-warning/10 text-warning',
                  )}
                >
                  {improving ? '↓' : '↑'} {Math.abs(Number(pctChange))}%
                </span>
              )
            })()}
        </div>
      </CardHeader>
      <CardContent>
        <KpiGrid columns={job.reward_history?.length ? 4 : 3}>
          {job.loss != null && <StatCard label="Final loss" value={job.loss.toFixed(4)} />}
          {job.train_loss != null && (
            <StatCard label="Train loss" value={job.train_loss.toFixed(4)} />
          )}
          {job.eval_loss != null && (
            <StatCard label="Validation loss" value={job.eval_loss.toFixed(4)} />
          )}
          {typeof job.result?.final_reward === 'number' && (
            <StatCard label="Final reward" value={job.result.final_reward.toFixed(4)} />
          )}
        </KpiGrid>
        {job.loss_history && job.loss_history.length > 1 && (
          <div className="mt-4">
            <LossChart
              data={job.loss_history.map(
                (p) => ({ step: p.step, value: p.value, type: p.type }) as LossPoint,
              )}
              rewardData={job.reward_history?.map(
                (p) => ({ step: p.step, value: p.value }) as RewardPoint,
              )}
              live={job.status === 'running'}
            />
          </div>
        )}
      </CardContent>
    </Card>
  )
}
