'use client'

import { memo, useMemo } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { Checkpoint } from '@/lib/souls-controller'

interface TrainingTipsCardProps {
  checkpoints: Checkpoint[]
  loading?: boolean
}

interface TrainingTip {
  message: string
  type: 'info' | 'warning' | 'success'
}

function generateTips(checkpoints: Checkpoint[]): TrainingTip[] {
  const tips: TrainingTip[] = []
  
  if (checkpoints.length === 0) {
    tips.push({ message: 'Start training to see personalized tips and recommendations.', type: 'info' })
    return tips
  }

  const withLoss = checkpoints.filter(c => c.loss != null && c.loss > 0)
  const withQuality = checkpoints.filter(c => c.avg_quality != null && c.avg_quality > 0)

  // Dataset size tips
  if (checkpoints.length < 5) {
    tips.push({ message: 'Small number of checkpoints. Train more to get better health analysis.', type: 'info' })
  }

  // Loss analysis
  if (withLoss.length >= 2) {
    const losses = withLoss.map(c => c.loss!)
    const bestLoss = Math.min(...losses)
    const latestLoss = losses[0]
    const gap = latestLoss - bestLoss

    if (gap > 0.5) {
      tips.push({ message: `Large gap between current (${latestLoss.toFixed(4)}) and best (${bestLoss.toFixed(4)}) loss. Consider lowering learning rate.`, type: 'warning' })
    } else if (gap < 0.01 && withLoss.length > 5) {
      tips.push({ message: 'Training has converged. Consider stopping or reducing learning rate for fine-tuning.', type: 'success' })
    }

    // Check for overfitting (loss increasing while training continues)
    if (losses.length >= 3) {
      const recentTrend = losses[0] - losses[Math.min(2, losses.length - 1)]
      if (recentTrend > 0.1) {
        tips.push({ message: 'Loss trending upward. This may indicate overfitting. Try more data or lower learning rate.', type: 'warning' })
      }
    }
  }

  // Quality tips
  if (withQuality.length > 0) {
    const avgQuality = withQuality.reduce((s, c) => s + c.avg_quality!, 0) / withQuality.length
    if (avgQuality < 2.5) {
      tips.push({ message: `Low average quality (${avgQuality.toFixed(1)}/5). Consider curating higher quality training data.`, type: 'warning' })
    } else if (avgQuality > 4.0) {
      tips.push({ message: `Excellent data quality (${avgQuality.toFixed(1)}/5). Keep up the good work!`, type: 'success' })
    }
  }

  // Checkpoint tips
  if (checkpoints.length > 10) {
    tips.push({ message: 'Many checkpoints saved. Consider purging old ones to save disk space.', type: 'info' })
  }

  // Default tip if nothing else
  if (tips.length === 0) {
    tips.push({ message: 'Training looks healthy. Keep going!', type: 'success' })
  }

  return tips
}

const TIP_STYLES: Record<TrainingTip['type'], { icon: string; bg: string }> = {
  info: { icon: 'ℹ️', bg: 'bg-muted/50' },
  warning: { icon: '⚠️', bg: 'bg-warning/5' },
  success: { icon: '✓', bg: 'bg-success/5' },
}

export const TrainingTipsCard = memo(function TrainingTipsCard({ checkpoints, loading }: TrainingTipsCardProps) {
  const tips = useMemo(() => generateTips(checkpoints), [checkpoints])

  if (loading && checkpoints.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Training tips</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="h-4 w-48 animate-pulse rounded bg-muted" />
            <div className="h-3 w-32 animate-pulse rounded bg-muted" />
          </div>
        </CardContent>
      </Card>
    )
  }

  if (tips.length === 0) return null

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Training tips</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {tips.map((tip, i) => {
            const styles = TIP_STYLES[tip.type]
            return (
              <div
                key={i}
                className={`flex items-start gap-2 text-xs p-2 rounded ${styles.bg}`}
              >
                <span className="shrink-0 mt-0.5">{styles.icon}</span>
                <span className="text-muted-foreground">{tip.message}</span>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
})
