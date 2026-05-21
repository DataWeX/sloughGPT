'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'

export function SelfTrainToggle() {
  const [status, setStatus] = useState<{running?: boolean; steps?: number} | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch('/auto-train/status').then(r => r.json()).then(setStatus).catch(() => {})
    const interval = setInterval(() => {
      fetch('/auto-train/status').then(r => r.json()).then(setStatus).catch(() => {})
    }, 5000)
    return () => clearInterval(interval)
  }, [])

  const isRunning = status?.running === true
  const steps = status?.steps || 0

  const toggle = async () => {
    setLoading(true)
    try {
      await fetch(isRunning ? '/auto-train/stop' : '/auto-train/start', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' })
      const res = await fetch('/auto-train/status')
      setStatus(await res.json())
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-3">
        {isRunning && <span className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />}
        <div>
          <div className="font-medium">{isRunning ? 'Learning...' : 'Ready'}</div>
          <div className="text-sm text-muted-foreground">{steps} steps learned</div>
        </div>
      </div>
      <Button onClick={toggle} disabled={loading} variant={isRunning ? 'destructive' : 'default'} size="sm">
        {loading ? '...' : isRunning ? 'Stop' : 'Start'}
      </Button>
    </div>
  )
}
