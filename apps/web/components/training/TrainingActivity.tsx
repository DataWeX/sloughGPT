'use client'

import { useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingActivityProps {
  checkpoints: Checkpoint[]
  maxItems?: number
}

interface ActivityItem {
  id: string
  icon: string
  text: string
  time?: string
  variant: 'default' | 'success' | 'warning'
}

function timeAgo(ts?: string): string {
  if (!ts) return ''
  try {
    const d = new Date(ts)
    if (isNaN(d.getTime())) return ''
    const diffMs = Date.now() - d.getTime()
    const diffM = Math.floor(diffMs / 60000)
    const diffH = Math.floor(diffM / 60)
    const diffD = Math.floor(diffH / 24)
    if (diffD > 0) return `${diffD}d ago`
    if (diffH > 0) return `${diffH}h ago`
    if (diffM > 0) return `${diffM}m ago`
    return 'just now'
  } catch {
    return ''
  }
}

function buildActivity(checkpoints: Checkpoint[], maxItems: number): ActivityItem[] {
  const items: ActivityItem[] = []

  for (const c of checkpoints) {
    if (c.is_loaded) {
      items.push({
        id: `load-${c.name}`,
        icon: '\u25B6',
        text: `Loaded ${c.name}`,
        time: timeAgo(c.born_at),
        variant: 'success',
      })
    }

    if (c.verdict === 'overfit') {
      items.push({
        id: `overfit-${c.name}`,
        icon: '\u26A0',
        text: `${c.name} overfitting`,
        time: timeAgo(c.born_at),
        variant: 'warning',
      })
    }

    if (c.loss != null && c.loss > 0) {
      items.push({
        id: `cp-${c.name}`,
        icon: '\u2022',
        text: `Saved ${c.name} (${c.loss.toFixed(3)})`,
        time: timeAgo(c.born_at),
        variant: 'default',
      })
    }
  }

  items.sort((a, b) => {
    if (!a.time && !b.time) return 0
    if (!a.time) return 1
    if (!b.time) return -1
    return 0
  })

  return items.slice(0, maxItems)
}

const variantDot: Record<string, string> = {
  default: 'bg-muted-foreground/40',
  success: 'bg-success',
  warning: 'bg-warning',
}

export function TrainingActivity({ checkpoints, maxItems = 5 }: TrainingActivityProps) {
  const items = useMemo(() => buildActivity(checkpoints, maxItems), [checkpoints, maxItems])

  if (items.length === 0) return null

  return (
    <Card data-testid="training-activity">
      <CardHeader className="flex flex-row items-center justify-between py-3">
        <CardTitle className="text-base">Recent Activity</CardTitle>
        <Badge variant="secondary" className="text-[10px] px-1.5 py-0 h-4">
          {items.length}
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {items.map(item => (
            <div key={item.id} className="flex items-start gap-2 text-[11px]">
              <span className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${variantDot[item.variant]}`} />
              <div className="flex-1 min-w-0">
                <span className="truncate">{item.text}</span>
              </div>
              {item.time && (
                <span className="text-[10px] text-muted-foreground/50 shrink-0">{item.time}</span>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
