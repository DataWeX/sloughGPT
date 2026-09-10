'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

interface FeedbackEntry {
  timestamp: string
  rating: 'thumbs_up' | 'thumbs_down'
  quality_score?: number
}

interface SentimentTrendCardProps {
  history: FeedbackEntry[]
}

function SimpleSparkline({ values, color, height = 40 }: { values: number[]; color: string; height?: number }) {
  const points = useMemo(() => {
    if (values.length === 0) return ''
    const max = Math.max(...values)
    const min = Math.min(...values)
    const range = max - min || 1
    const step = 100 / Math.max(values.length - 1, 1)
    return values.map((v, i) => `${i * step},${100 - ((v - min) / range) * 100}`).join(' ')
  }, [values])

  if (!points) return <div className="flex items-center justify-center text-[9px] text-muted-foreground" style={{ height }}>No data</div>

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full" style={{ height }} role="img" aria-label="Trend chart">
      <polyline fill="none" stroke={color} strokeWidth="1.5" points={points} />
    </svg>
  )
}

function BucketsToSparkline({ entries, bucketCount = 20 }: { entries: FeedbackEntry[]; bucketCount?: number }) {
  const buckets = useMemo(() => {
    if (entries.length === 0) return []
    const sorted = [...entries].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    const bucketSize = Math.max(1, Math.ceil(sorted.length / bucketCount))
    const result: number[] = []
    for (let i = 0; i < sorted.length; i += bucketSize) {
      const slice = sorted.slice(i, i + bucketSize)
      const ups = slice.filter(e => e.rating === 'thumbs_up').length
      result.push(ups / slice.length)
    }
    return result
  }, [entries, bucketCount])

  return <SimpleSparkline values={buckets} color="#8b5cf6" />
}

function formatTimeAgo(ts: string): string {
  try {
    const diff = Date.now() - new Date(ts).getTime()
    const mins = Math.floor(diff / 60000)
    if (mins < 60) return `${mins}m ago`
    const hours = Math.floor(mins / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  } catch {
    return ts
  }
}

export function SentimentTrendCard({ history }: SentimentTrendCardProps) {
  const stats = useMemo(() => {
    if (history.length === 0) return null
    const ups = history.filter(e => e.rating === 'thumbs_up').length
    const downs = history.filter(e => e.rating === 'thumbs_down').length
    const ratio = history.length > 0 ? ups / history.length : 0

    const last7 = history.filter(e => {
      const diff = Date.now() - new Date(e.timestamp).getTime()
      return diff < 7 * 24 * 60 * 60 * 1000
    })
    const recentRatio = last7.length > 0 ? last7.filter(e => e.rating === 'thumbs_up').length / last7.length : ratio

    const trend = recentRatio > ratio ? 'improving' : recentRatio < ratio ? 'declining' : 'stable'

    return { ups, downs, ratio, recentRatio, trend, total: history.length, last7Count: last7.length }
  }, [history])

  if (history.length === 0) {
    return (
      <Card data-testid="sentiment-trend">
        <CardHeader><CardTitle className="text-base">Sentiment Trend</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground text-center py-3">No feedback history yet.</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="sentiment-trend">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Sentiment Trend</CardTitle>
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
            stats!.trend === 'improving' ? 'bg-success/15 text-success' :
            stats!.trend === 'declining' ? 'bg-destructive/15 text-destructive' :
            'bg-muted text-muted-foreground'
          }`}>
            {stats!.trend === 'improving' ? '↑' : stats!.trend === 'declining' ? '↓' : '→'} {stats!.trend}
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-4 gap-2">
          {[
            { label: 'Total', value: String(stats!.total) },
            { label: 'Positive', value: String(stats!.ups) },
            { label: 'Negative', value: String(stats!.downs) },
            { label: '7d Count', value: String(stats!.last7Count) },
          ].map(s => (
            <div key={s.label} className="text-center p-1.5 rounded bg-muted/30">
              <p className="text-[9px] text-muted-foreground">{s.label}</p>
              <p className="text-[10px] font-mono font-medium">{s.value}</p>
            </div>
          ))}
        </div>

        <div>
          <p className="text-[10px] text-muted-foreground mb-1">Overall sentiment ratio</p>
          <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${stats!.ratio * 100}%`, backgroundColor: stats!.ratio >= 0.6 ? '#22c55e' : stats!.ratio >= 0.4 ? '#f59e0b' : '#ef4444' }}
            />
          </div>
          <p className="text-[9px] text-muted-foreground mt-0.5">{(stats!.ratio * 100).toFixed(1)}% positive</p>
        </div>

        <div>
          <p className="text-[10px] text-muted-foreground mb-1">Sentiment over time</p>
          <div className="rounded-md bg-muted/20 p-2">
            <BucketsToSparkline entries={history} />
          </div>
        </div>

        <div>
          <p className="text-[10px] text-muted-foreground mb-1">Recent feedback</p>
          <div className="space-y-1">
            {history.slice(-5).reverse().map((entry, i) => (
              <div key={i} className="flex items-center justify-between text-[10px] py-0.5 border-b border-border/20 last:border-0">
                <div className="flex items-center gap-1.5">
                  <span className={entry.rating === 'thumbs_up' ? 'text-success' : 'text-destructive'}>
                    {entry.rating === 'thumbs_up' ? '👍' : '👎'}
                  </span>
                  {entry.quality_score != null && (
                    <span className="text-muted-foreground">Score: {(entry.quality_score * 100).toFixed(0)}%</span>
                  )}
                </div>
                <span className="text-muted-foreground/60">{formatTimeAgo(entry.timestamp)}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
