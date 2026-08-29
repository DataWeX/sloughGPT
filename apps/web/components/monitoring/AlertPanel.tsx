'use client'

import { memo } from 'react'
import { Card, CardContent, Slider } from '@sloughgpt/strui'

interface AlertPanelProps {
  cpuThreshold: number
  memThreshold: number
  onCpuThresholdChange: (v: number) => void
  onMemThresholdChange: (v: number) => void
  alerts: Array<{ time: string; type: string; value: number }>
}

export const AlertPanel = memo(function AlertPanel({ cpuThreshold, memThreshold, onCpuThresholdChange, onMemThresholdChange, alerts }: AlertPanelProps) {
  const hasNotificationAPI = typeof window !== 'undefined' && 'Notification' in window

  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Alert thresholds</span>
        {hasNotificationAPI && (
          <button
            type="button"
            onClick={() => {
              if (Notification.permission === 'granted') {
                new Notification('Notifications enabled', { body: 'You will be notified when thresholds are breached.' })
              } else {
                Notification.requestPermission()
              }
            }}
            className="text-[10px] text-muted-foreground hover:text-foreground transition-colors"
          >
            {Notification.permission === 'granted' ? 'Notifications on' : 'Enable notifications'}
          </button>
        )}
      </div>
      <CardContent className="p-0 space-y-2">
        <Slider
          id="cpu-threshold"
          label="CPU"
          value={[cpuThreshold]}
          min={10}
          max={100}
          showValue
          formatValue={(v) => `${v}%`}
          onValueChange={([v]) => onCpuThresholdChange(v)}
          size="sm"
        />
        <Slider
          id="mem-threshold"
          label="MEM"
          value={[memThreshold]}
          min={10}
          max={100}
          showValue
          formatValue={(v) => `${v}%`}
          onValueChange={([v]) => onMemThresholdChange(v)}
          size="sm"
        />
      </CardContent>

      {alerts.length > 0 && (
        <>
          <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mt-3 mb-1 block">Recent alerts</span>
          <div className="space-y-1 max-h-24 overflow-y-auto" role="log" aria-live="polite" aria-label="Recent alerts">
            {alerts.slice(0, 5).map((alert, i) => (
              <div key={i} className="flex items-center justify-between text-[10px]">
                <span className="text-warning">{alert.type} {alert.value.toFixed(0)}%</span>
                <span className="text-muted-foreground">{alert.time}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </Card>
  )
})
