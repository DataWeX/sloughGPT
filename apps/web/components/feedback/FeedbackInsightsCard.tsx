'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
import type { FeedbackStats } from '@/lib/feedback-controller'

interface FeedbackInsightsCardProps {
  stats: FeedbackStats | null
}

export function FeedbackInsightsCard({ stats }: FeedbackInsightsCardProps) {
  if (!stats?.db_stats) return null

  const { db_stats } = stats
  const { thumbs_up, thumbs_down, feedback_total, conversations, messages, ratio } = db_stats

  const feedbackPerConv = conversations > 0 ? (feedback_total / conversations).toFixed(1) : '0'
  const feedbackPerMsg = messages > 0 ? (feedback_total / messages).toFixed(2) : '0'
  const sentimentScore = Math.round(ratio * 100)

  const qualityLabel = ratio >= 0.8 ? 'Excellent' : ratio >= 0.6 ? 'Good' : ratio >= 0.4 ? 'Fair' : 'Poor'
  const qualityColor = ratio >= 0.8 ? 'text-success' : ratio >= 0.6 ? 'text-primary' : ratio >= 0.4 ? 'text-warning' : 'text-destructive'

  const activityLevel = stats.history_length > 100 ? 'High' : stats.history_length > 20 ? 'Medium' : 'Low'
  const activityColor = stats.history_length > 100 ? 'text-success' : stats.history_length > 20 ? 'text-primary' : 'text-muted-foreground'

  return (
    <Card data-testid="feedback-insights">
      <CardHeader>
        <CardTitle className="text-base">Feedback Insights</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-2 mb-3">
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Sentiment</div>
            <div className={`text-sm font-mono font-medium ${qualityColor}`}>{sentimentScore}%</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Quality</div>
            <div className={`text-sm font-mono font-medium ${qualityColor}`}>{qualityLabel}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Fb/Conv</div>
            <div className="text-sm font-mono font-medium">{feedbackPerConv}</div>
          </div>
          <div className="rounded-md bg-muted/30 p-2 text-center">
            <div className="text-[10px] text-muted-foreground">Activity</div>
            <div className={`text-sm font-mono font-medium ${activityColor}`}>{activityLevel}</div>
          </div>
        </div>

        <div className="space-y-1.5 text-[11px]">
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Feedback per message</span>
            <span className="font-mono">{feedbackPerMsg}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Total conversations</span>
            <span className="font-mono">{conversations}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">Total messages</span>
            <span className="font-mono">{messages}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-muted-foreground">History entries</span>
            <span className="font-mono">{stats.history_length}</span>
          </div>
          {stats.current_weights && (
            <>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Temperature</span>
                <span className="font-mono">{stats.current_weights.temperature}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Repetition penalty</span>
                <span className="font-mono">{stats.current_weights.repetition_penalty}</span>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
