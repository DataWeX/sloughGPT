'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input } from '@sloughgpt/strui'

interface AutoTrainConfigCardProps {
  threshold?: number
  intervalS?: number
  onSave?: (threshold: number, intervalS: number) => Promise<void>
}

export function AutoTrainConfigCard({ threshold: initThreshold = 10, intervalS: initInterval = 120, onSave }: AutoTrainConfigCardProps) {
  const [threshold, setThreshold] = useState(initThreshold)
  const [interval, setInterval] = useState(initInterval)
  const [saving, setSaving] = useState(false)

  const handleSave = async () => {
    if (!onSave) return
    setSaving(true)
    try { await onSave(threshold, interval) } finally { setSaving(false) }
  }

  return (
    <Card data-testid="auto-train-config">
      <CardHeader>
        <CardTitle className="text-base">Configuration</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Pair Threshold</div>
            <Input
              type="number"
              min={10}
              max={10000}
              value={threshold}
              onChange={e => setThreshold(parseInt(e.target.value) || 10)}
              data-testid="threshold-input"
            />
            <div className="text-[10px] text-muted-foreground mt-0.5">Min pairs before auto-training triggers</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium mb-1">Check Interval (seconds)</div>
            <Input
              type="number"
              min={30}
              max={3600}
              value={interval}
              onChange={e => setInterval(parseInt(e.target.value) || 120)}
              data-testid="interval-input"
            />
            <div className="text-[10px] text-muted-foreground mt-0.5">How often to check for new training data</div>
          </div>
          {onSave && (
            <Button size="sm" onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : 'Save Configuration'}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
