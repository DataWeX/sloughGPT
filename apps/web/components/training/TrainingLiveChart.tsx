'use client'

import { memo, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle, StatCard, KpiGrid } from '@sloughgpt/strui'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

interface LossPoint {
  step: number
  loss: number
  isEval?: boolean
}

interface TrainingLiveChartProps {
  lossHistory: LossPoint[]
  progress: number
  epoch: number
  totalEpochs: number
  globalStep: number
  totalSteps: number
  stepsPerSec: number | null
  eta: number | null
}

function LossTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ name: string; value: number; color: string }>; label?: string }) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-card border border-border rounded-lg px-3 py-2 shadow-lg text-xs font-numeric">
      <p className="text-muted-foreground mb-1">Step {label}</p>
      {payload.map((entry, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: entry.color }} />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-medium">{entry.value.toFixed(4)}</span>
        </div>
      ))}
    </div>
  )
}

function formatEta(seconds: number | null): string {
  if (seconds == null || seconds <= 0) return '—'
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

export const TrainingLiveChart = memo(function TrainingLiveChart({
  lossHistory,
  progress,
  epoch,
  totalEpochs,
  globalStep,
  totalSteps,
  stepsPerSec,
  eta,
}: TrainingLiveChartProps) {
  const chartData = useMemo(() => {
    if (lossHistory.length === 0) return []
    return lossHistory.map(p => ({
      step: p.step,
      loss: p.loss,
      type: p.isEval ? 'eval' : 'train',
    }))
  }, [lossHistory])

  const trainData = useMemo(() => chartData.filter(d => d.type === 'train'), [chartData])
  const evalData = useMemo(() => chartData.filter(d => d.type === 'eval'), [chartData])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Training Progress</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <KpiGrid columns={4}>
          <StatCard label="Progress" value={`${(progress * 100).toFixed(1)}%`} />
          <StatCard label="Epoch" value={totalEpochs > 0 ? `${epoch}/${totalEpochs}` : `${epoch}`} />
          <StatCard label="Step" value={totalSteps > 0 ? `${globalStep}/${totalSteps}` : `${globalStep}`} />
          <StatCard label="ETA" value={formatEta(eta)} />
        </KpiGrid>

        {chartData.length > 1 ? (
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis
                  dataKey="step"
                  tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
                  tickLine={false}
                  axisLine={false}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
                  width={45}
                  tickLine={false}
                  axisLine={false}
                  domain={['auto', 'auto']}
                />
                <Tooltip content={<LossTooltip />} />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="var(--primary)"
                  strokeWidth={1.5}
                  dot={false}
                  name="Loss"
                />
                {evalData.length > 0 && (
                  <Line
                    data={evalData}
                    type="monotone"
                    dataKey="loss"
                    stroke="var(--warning)"
                    strokeWidth={1.5}
                    dot={{ r: 2, fill: 'var(--warning)' }}
                    name="Eval Loss"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-48 flex items-center justify-center text-xs text-muted-foreground">
            Waiting for loss data...
          </div>
        )}

        {stepsPerSec != null && stepsPerSec > 0 && (
          <div className="text-[10px] text-muted-foreground text-right">
            {stepsPerSec.toFixed(1)} steps/s
          </div>
        )}
      </CardContent>
    </Card>
  )
})
