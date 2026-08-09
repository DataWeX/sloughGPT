'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'
import { selfTrainController, type SelfTrainStatus } from '@/lib/self-train-controller'

export function SelfTrainCard() {
  const [status, setStatus] = useState<SelfTrainStatus | null>(null)
  const [model, setModel] = useState('')
  const [temperature, setTemperature] = useState('0.8')
  const [forever, setForever] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = useCallback(async () => {
    try {
      const data = await selfTrainController.getStatus()
      setStatus(data)
    } catch { /* offline */ }
  }, [])

  useEffect(() => {
    fetchStatus()
    const iv = setInterval(fetchStatus, 5000)
    return () => clearInterval(iv)
  }, [fetchStatus])

  const handleStart = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await selfTrainController.start({
        model: model.trim() || undefined,
        temperature: temperature ? parseFloat(temperature) : undefined,
        forever,
      })
      if (result?.status === 'error') {
        setError(result.error ?? 'Unknown error')
      } else {
        await fetchStatus()
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed')
    } finally {
      setLoading(false)
    }
  }

  const handleStop = async () => {
    setLoading(true)
    try {
      await selfTrainController.stop()
      await fetchStatus()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed')
    } finally {
      setLoading(false)
    }
  }

  const isRunning = status?.status === 'running'

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Self-Train</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <Input value={model} onChange={e => setModel(e.target.value)} placeholder="Model (optional)" />
          <Input value={temperature} onChange={e => setTemperature(e.target.value)} placeholder="Temperature" type="number" step="0.1" min="0" max="2" />
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={forever} onChange={e => setForever(e.target.checked)} className="accent-primary" />
          Train indefinitely
        </label>
        {error && <div className="text-xs text-destructive">{error}</div>}
        <div className="flex items-center gap-3">
          <Button size="sm" onClick={handleStart} disabled={loading || isRunning}>
            {isRunning ? 'Running...' : 'Start'}
          </Button>
          <Button size="sm" variant="outline" onClick={handleStop} disabled={loading || !isRunning}>
            Stop
          </Button>
          {isRunning && <span className="text-xs text-muted-foreground">PID {status?.pid}</span>}
        </div>
        {status?.history && status.history.length > 0 && (
          <div className="max-h-32 overflow-y-auto font-mono text-xs space-y-0.5">
            {status.history.slice(-10).map((line, i) => (
              <div key={i} className="text-muted-foreground">{line}</div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
