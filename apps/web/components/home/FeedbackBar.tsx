'use client'

import Link from 'next/link'
import { Card, CardContent, cn } from '@sloughgpt/strui'
import { IconThumbUp, IconThumbDown } from '@sloughgpt/strui'
import type { FeedbackStats } from '@/lib/feedback-controller'

interface FeedbackBarProps {
  loading: boolean
  feedbackStats: FeedbackStats | null
}

export function FeedbackBar({ loading, feedbackStats }: FeedbackBarProps) {
  if (loading) {
    return (
      <Card>
        <CardContent className="py-3">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="h-4 w-24 animate-pulse rounded bg-muted" />
            <div className="h-4 w-16 animate-pulse rounded bg-muted" />
            <div className="h-4 w-20 animate-pulse rounded bg-muted" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!feedbackStats?.db_stats || feedbackStats.db_stats.feedback_total === 0) return null

  const { thumbs_up, thumbs_down, ratio, feedback_total } = feedbackStats.db_stats
  const positivePercent = Math.round(ratio * 100)

  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-muted-foreground">Feedback</span>
            <span className="text-sm font-semibold tabular-nums">{feedback_total}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="inline-flex items-center gap-1 text-xs">
              <span className="text-success"><IconThumbUp className="h-3 w-3" /></span>
              <span className="font-medium tabular-nums">{thumbs_up}</span>
            </span>
            <span className="inline-flex items-center gap-1 text-xs">
              <span className="text-destructive"><IconThumbDown className="h-3 w-3" /></span>
              <span className="font-medium tabular-nums">{thumbs_down}</span>
            </span>
          </div>
          <div className="flex items-center gap-2 flex-1 min-w-[140px]">
            <div className="flex-1 h-1.5 rounded-full bg-muted/40 overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-500",
                  ratio >= 0.7 ? 'bg-success' : ratio >= 0.4 ? 'bg-warning' : 'bg-destructive'
                )}
                style={{ width: `${positivePercent}%` }}
              />
            </div>
            <span className={cn(
              "text-xs font-medium tabular-nums shrink-0",
              ratio >= 0.7 ? 'text-success' : ratio >= 0.4 ? 'text-warning' : 'text-destructive'
            )}>
              {positivePercent}%
            </span>
          </div>
          <Link href="/training" className="text-xs text-primary hover:text-primary/80 shrink-0">
            Train from feedback →
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}
