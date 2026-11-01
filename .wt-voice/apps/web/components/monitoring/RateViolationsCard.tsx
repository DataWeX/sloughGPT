'use client'

import { memo } from 'react'
import { Card, CardContent } from '@sloughgpt/strui'
import type { LiveHealthSnapshot } from '@/hooks/useLiveStatus'
import { useTick } from '@/hooks/useTick'
import { timeAgo } from '@/lib/time-ago'

interface RateViolationsCardProps {
  liveHealth: LiveHealthSnapshot | null
}

export const RateViolationsCard = memo(function RateViolationsCard({ liveHealth }: RateViolationsCardProps) {
  const violations = liveHealth?.rate_violations ?? []
  useTick()

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
})
