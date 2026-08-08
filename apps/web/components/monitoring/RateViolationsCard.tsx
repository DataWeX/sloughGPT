'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'

interface RateViolationsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() / 1000 - ts))
  if (s < 5) return 'just now'
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  return `${Math.floor(m / 60)}h ago`
}

export function RateViolationsCard({ liveHealth }: RateViolationsCardProps) {
  const violations = liveHealth?.rate_violations ?? []
  const [tick, setTick] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 10_000)
    return () => clearInterval(id)
  }, [])

  if (violations.length === 0) return null

  return (
    <Card className="p-3">
      <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2 block">Rate violations</span>
      <CardContent className="p-0 max-h-[220px] overflow-y-auto space-y-1.5">
        {violations.map((v, i) => (
          <div key={`${v.ts}-${i}`} className="border border-warning/25 hover:bg-warning/5 transition-colors rounded-md p-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium truncate font-mono">{v.path}</span>
              <span className="shrink-0 text-[9px] px-1.5 py-0.5 rounded font-medium bg-warning/15 text-warning tabular-nums">
                {v.count}/{v.limit}/s
              </span>
            </div>
            <div className="text-[10px] opacity-60 mt-0.5 font-mono">{timeAgo(v.ts)}</div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
