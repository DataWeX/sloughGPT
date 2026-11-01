'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface ExperimentLogCardProps {
  experimentId?: string | null
  onLogMetric?: (id: string, name: string, value: number) => Promise<void>
  onLogParam?: (id: string, name: string, value: string) => Promise<void>
  onComplete?: (id: string) => Promise<void>
}

export function ExperimentLogCard({ experimentId, onLogMetric, onLogParam, onComplete }: ExperimentLogCardProps) {
  const [metricName, setMetricName] = useState('')
  const [metricValue, setMetricValue] = useState('')
  const [paramName, setParamName] = useState('')
  const [paramValue, setParamValue] = useState('')
  const [loading, setLoading] = useState<'metric' | 'param' | 'complete' | null>(null)

  if (!experimentId) return null

  const handleLogMetric = async () => {
    if (!metricName.trim() || !metricValue.trim() || !onLogMetric) return
    setLoading('metric')
    try {
      await onLogMetric(experimentId, metricName.trim(), parseFloat(metricValue))
      setMetricName('')
      setMetricValue('')
    } finally { setLoading(null) }
  }

  const handleLogParam = async () => {
    if (!paramName.trim() || !paramValue.trim() || !onLogParam) return
    setLoading('param')
    try {
      await onLogParam(experimentId, paramName.trim(), paramValue.trim())
      setParamName('')
      setParamValue('')
    } finally { setLoading(null) }
  }

  const handleComplete = async () => {
    if (!onComplete) return
    setLoading('complete')
    try { await onComplete(experimentId) } finally { setLoading(null) }
  }

  return (
    <Card data-testid="experiment-log">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Log Data</CardTitle>
          {onComplete && (
            <Button size="sm" variant="ghost" className="text-success text-[10px]" onClick={handleComplete} disabled={loading === 'complete'}>
              {loading === 'complete' ? 'Completing...' : 'Complete'}
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {onLogMetric && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Metric</div>
              <div className="flex gap-1.5">
                <Input
                  value={metricName}
                  onChange={e => setMetricName(e.target.value)}
                  placeholder="name"
                  className="flex-1"
                  data-testid="metric-name"
                />
                <Input
                  value={metricValue}
                  onChange={e => setMetricValue(e.target.value)}
                  placeholder="value"
                  type="number"
                  className="w-20"
                  data-testid="metric-value"
                />
                <Button size="sm" onClick={handleLogMetric} disabled={loading === 'metric' || !metricName.trim() || !metricValue.trim()}>
                  Log
                </Button>
              </div>
            </div>
          )}
          {onLogParam && (
            <div>
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Parameter</div>
              <div className="flex gap-1.5">
                <Input
                  value={paramName}
                  onChange={e => setParamName(e.target.value)}
                  placeholder="name"
                  className="flex-1"
                  data-testid="param-name"
                />
                <Input
                  value={paramValue}
                  onChange={e => setParamValue(e.target.value)}
                  placeholder="value"
                  className="flex-1"
                  data-testid="param-value"
                />
                <Button size="sm" onClick={handleLogParam} disabled={loading === 'param' || !paramName.trim() || !paramValue.trim()}>
                  Log
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
