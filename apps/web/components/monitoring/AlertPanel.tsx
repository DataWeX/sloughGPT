'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'

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
        <div className="flex items-center gap-2">
          <label htmlFor="cpu-threshold" className="text-[11px] text-muted-foreground w-10">CPU</label>
          <input id="cpu-threshold" type="range" min={10} max={100} value={cpuThreshold} onChange={e => onCpuThresholdChange(Number(e.target.value))} className="flex-1 h-1 accent-primary" />
          <span className="text-[11px] font-numeric w-8 text-right">{cpuThreshold}%</span>
        </div>
        <div className="flex items-center gap-2">
          <label htmlFor="mem-threshold" className="text-[11px] text-muted-foreground w-10">MEM</label>
          <input id="mem-threshold" type="range" min={10} max={100} value={memThreshold} onChange={e => onMemThresholdChange(Number(e.target.value))} className="flex-1 h-1 accent-primary" />
          <span className="text-[11px] font-numeric w-8 text-right">{memThreshold}%</span>
        </div>
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
