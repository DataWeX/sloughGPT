'use client'

import { InsightsCard } from '@sloughgpt/strui'
import type { FeedbackStats } from '@/lib/feedback-controller'

interface FeedbackInsightsCardProps {
  stats: FeedbackStats | null
}

export function FeedbackInsightsCard({ stats }: FeedbackInsightsCardProps) {
  if (!stats?.db_stats) return null

  const { db_stats } = stats
  const { feedback_total, conversations, messages, ratio } = db_stats

  const feedbackPerConv = conversations > 0 ? (feedback_total / conversations).toFixed(1) : '0'
  const feedbackPerMsg = messages > 0 ? (feedback_total / messages).toFixed(2) : '0'
  const sentimentScore = Math.round(ratio * 100)

  const qualityLabel = ratio >= 0.8 ? 'Excellent' : ratio >= 0.6 ? 'Good' : ratio >= 0.4 ? 'Fair' : 'Poor'
  const qualityColor = ratio >= 0.8 ? 'text-success' : ratio >= 0.6 ? 'text-primary' : ratio >= 0.4 ? 'text-warning' : 'text-destructive'

  const activityLevel = stats.history_length > 100 ? 'High' : stats.history_length > 20 ? 'Medium' : 'Low'
  const activityColor = stats.history_length > 100 ? 'text-success' : stats.history_length > 20 ? 'text-primary' : 'text-muted-foreground'

  return (
    <InsightsCard
      title="Feedback Insights"
      testId="feedback-insights"
      kpis={[
        { label: 'Sentiment', value: <span className={qualityColor}>{sentimentScore}%</span> },
        { label: 'Quality', value: <span className={qualityColor}>{qualityLabel}</span> },
        { label: 'Fb/Conv', value: feedbackPerConv },
        { label: 'Activity', value: <span className={activityColor}>{activityLevel}</span> },
      ]}
      kpiColumns={4}
      details={[
        { label: 'Feedback per message', value: feedbackPerMsg },
        { label: 'Total conversations', value: conversations },
        { label: 'Total messages', value: messages },
        { label: 'History entries', value: stats.history_length },
        ...(stats.current_weights
          ? [
              { label: 'Temperature', value: stats.current_weights.temperature },
              { label: 'Repetition penalty', value: stats.current_weights.repetition_penalty },
            ]
          : []),
      ]}
    />
  )
}
