'use client'

import { Card, CardHeader, CardTitle, CardContent, StatCard, KpiGrid } from '@sloughgpt/strui'
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
        <KpiGrid columns={4} className="mb-3">
          <StatCard label="Sentiment" value={<span className={qualityColor}>{sentimentScore}%</span>} />
          <StatCard label="Quality" value={<span className={qualityColor}>{qualityLabel}</span>} />
          <StatCard label="Fb/Conv" value={feedbackPerConv} />
          <StatCard label="Activity" value={<span className={activityColor}>{activityLevel}</span>} />
        </KpiGrid>

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
