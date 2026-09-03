'use client'

import Link from 'next/link'
import { Card, CardContent, cn } from '@sloughgpt/strui'
import { IconThumbUp, IconThumbDown } from '@sloughgpt/strui'
import type { FeedbackStats } from '@/lib/feedback-controller'

interface FeedbackBarProps {
  feedbackStats: FeedbackStats
}

export function FeedbackBar({ feedbackStats }: FeedbackBarProps) {
  if (!feedbackStats?.db_stats || feedbackStats.db_stats.feedback_total === 0) return null

  return (
    <Card>
      <CardContent className="py-3">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5">
            <span className="text-sm text-muted-foreground">Feedback</span>
            <span className="text-sm font-medium tabular-nums">{feedbackStats.db_stats.feedback_total}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-sm text-success flex items-center gap-1"><IconThumbUp className="h-3.5 w-3.5" /> {feedbackStats.db_stats.thumbs_up}</span>
            <span className="text-sm text-destructive flex items-center gap-1"><IconThumbDown className="h-3.5 w-3.5" /> {feedbackStats.db_stats.thumbs_down}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-sm text-muted-foreground">Ratio</span>
            <span className={cn('text-sm font-medium', feedbackStats.db_stats.ratio >= 0.5 ? 'text-success' : 'text-warning')}>
              {Math.round(feedbackStats.db_stats.ratio * 100)}% positive
            </span>
          </div>
          <Link href="/training" className="ml-auto text-sm text-primary hover:text-primary/80 shrink-0">
            Train from feedback →
          </Link>
        </div>
      </CardContent>
    </Card>
  )
}
