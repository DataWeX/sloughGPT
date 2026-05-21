'use client'

import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function SelfTrainProgress() {
  const [logs, setLogs] = useState<string[]>([])

  useEffect(() => {
    fetch('/auto-train/log')
      .then(r => r.json())
      .then(d => setLogs(d.logs || []))
      .catch(() => {})
  }, [])

  if (logs.length === 0) return null

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Training Log</CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="h-48 overflow-y-auto space-y-1 text-sm font-mono">
          {logs.slice(-20).map((log, i) => (
            <div key={i} className="text-muted-foreground truncate">{log}</div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
