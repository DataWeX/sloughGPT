'use client'

import { useState, useEffect, useMemo, memo } from 'react'
import { Button, IconRefresh } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import { chatDB } from '@/lib/db'
import type { ChatSession } from '@/lib/db'

interface AnalyticsData {
  totalSessions: number
  totalMessages: number
  totalWords: number
  totalCharacters: number
  avgMessagesPerSession: number
  avgWordsPerMessage: number
  activeDays: number
  mostActiveDay: string
  sessionTrend: Array<{ date: string; count: number }>
  roleDistribution: { user: number; assistant: number }
}

interface ChatAnalyticsProps {
  className?: string
}

function computeAnalytics(sessions: ChatSession[]): AnalyticsData {
  if (sessions.length === 0) {
    return {
      totalSessions: 0,
      totalMessages: 0,
      totalWords: 0,
      totalCharacters: 0,
      avgMessagesPerSession: 0,
      avgWordsPerMessage: 0,
      activeDays: 0,
      mostActiveDay: 'N/A',
      sessionTrend: [],
      roleDistribution: { user: 0, assistant: 0 },
    }
  }

  let totalMessages = 0
  let totalWords = 0
  let totalCharacters = 0
  let userMessages = 0
  let assistantMessages = 0
  const dayCounts: Record<string, number> = {}

  for (const session of sessions) {
    const messages = session.messages || []
    totalMessages += messages.length

    for (const msg of messages) {
      totalCharacters += msg.content?.length || 0
      totalWords += (msg.content?.split(/\s+/) || []).filter(Boolean).length
      if (msg.role === 'user') userMessages++
      if (msg.role === 'assistant') assistantMessages++
    }

    const date = new Date(session.updatedAt || session.createdAt)
    const dayKey = date.toLocaleDateString('en-US', { weekday: 'long' })
    dayCounts[dayKey] = (dayCounts[dayKey] || 0) + 1
  }

  const mostActiveDay = Object.entries(dayCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || 'N/A'
  const activeDays = new Set(
    sessions.map(s => new Date(s.updatedAt || s.createdAt).toDateString())
  ).size

  const sessionTrend = sessions
    .sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime())
    .slice(-7)
    .map(s => ({
      date: new Date(s.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
      count: s.messages?.length || 0,
    }))

  return {
    totalSessions: sessions.length,
    totalMessages,
    totalWords,
    totalCharacters,
    avgMessagesPerSession: Math.round(totalMessages / sessions.length),
    avgWordsPerMessage: totalMessages > 0 ? Math.round(totalWords / totalMessages) : 0,
    activeDays,
    mostActiveDay,
    sessionTrend,
    roleDistribution: { user: userMessages, assistant: assistantMessages },
  }
}

function StatCard({ label, value, subtext }: { label: string; value: string | number; subtext?: string }) {
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{label}</div>
      <div className="text-sm font-medium">{typeof value === 'number' ? value.toLocaleString() : value}</div>
      {subtext && <div className="text-[10px] text-muted-foreground">{subtext}</div>}
    </div>
  )
}

function MiniBarChart({ data }: { data: Array<{ date: string; count: number }> }) {
  const max = Math.max(...data.map(d => d.count), 1)
  
  return (
    <div className="flex items-end gap-1 h-12">
      {data.map((d, i) => (
        <div key={i} className="flex-1 flex flex-col items-center gap-0.5">
          <div
            className="w-full bg-primary/20 rounded-t"
            style={{ height: `${(d.count / max) * 100}%`, minHeight: '2px' }}
          />
          <span className="text-[8px] text-muted-foreground">{d.date}</span>
        </div>
      ))}
    </div>
  )
}

export const ChatAnalytics = memo(function ChatAnalytics({ className }: ChatAnalyticsProps) {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(false)

  const loadAnalytics = async () => {
    setLoading(true)
    try {
      const sessions = await chatDB.loadSessions()
      setAnalytics(computeAnalytics(sessions))
    } catch {
      setAnalytics(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAnalytics()
  }, [])

  if (!analytics || analytics.totalSessions === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-2', className)}>
        {loading ? 'Loading analytics...' : 'No data available'}
      </div>
    )
  }

  return (
    <div className={cn('space-y-3', className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {analytics.activeDays} active days
        </span>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5"
          onClick={loadAnalytics}
          disabled={loading}
          aria-label="Refresh analytics"
        >
          <IconRefresh className={cn('h-3 w-3', loading && 'animate-spin')} />
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Sessions" value={analytics.totalSessions} />
        <StatCard label="Messages" value={analytics.totalMessages} />
        <StatCard label="Words" value={analytics.totalWords} />
        <StatCard label="Avg Msgs" value={analytics.avgMessagesPerSession} subtext="per session" />
      </div>

      {analytics.sessionTrend.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Recent Activity</div>
          <MiniBarChart data={analytics.sessionTrend} />
        </div>
      )}

      <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
        <span>Most active: {analytics.mostActiveDay}</span>
      </div>
    </div>
  )
})