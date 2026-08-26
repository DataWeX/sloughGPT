'use client'

import { useState, useMemo, useCallback, memo } from 'react'
import { Button, IconX, IconRefresh } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ChatAnalyticsDashboardProps {
  messages: ChatMessage[]
  className?: string
}

interface Analytics {
  totalMessages: number
  userMessages: number
  assistantMessages: number
  avgMessageLength: number
  longestMessage: number
  shortestMessage: number
  totalCharacters: number
  totalWords: number
  avgWordsPerMessage: number
  messagesByHour: number[]
  messagesByDay: number[]
  topWords: { word: string; count: number }[]
  responseTime: number
  conversationRate: number
}

function calculateAnalytics(messages: ChatMessage[]): Analytics {
  const userMessages = messages.filter(m => m.role === 'user')
  const assistantMessages = messages.filter(m => m.role === 'assistant')

  const lengths = messages.map(m => m.content.length)
  const words = messages.map(m => m.content.split(/\s+/).length)

  const messagesByHour = new Array(24).fill(0)
  const messagesByDay = new Array(7).fill(0)
  const wordCount: Record<string, number> = {}

  for (const msg of messages) {
    const date = new Date(msg.timestamp)
    messagesByHour[date.getHours()]++
    messagesByDay[date.getDay()]++

    const msgWords = msg.content.toLowerCase().split(/\s+/)
    for (const word of msgWords) {
      if (word.length > 3) {
        wordCount[word] = (wordCount[word] || 0) + 1
      }
    }
  }

  const topWords = Object.entries(wordCount)
    .map(([word, count]) => ({ word, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)

  return {
    totalMessages: messages.length,
    userMessages: userMessages.length,
    assistantMessages: assistantMessages.length,
    avgMessageLength: lengths.length > 0 ? Math.round(lengths.reduce((a, b) => a + b, 0) / lengths.length) : 0,
    longestMessage: lengths.length > 0 ? Math.max(...lengths) : 0,
    shortestMessage: lengths.length > 0 ? Math.min(...lengths) : 0,
    totalCharacters: lengths.reduce((a, b) => a + b, 0),
    totalWords: words.reduce((a, b) => a + b, 0),
    avgWordsPerMessage: words.length > 0 ? Math.round(words.reduce((a, b) => a + b, 0) / words.length) : 0,
    messagesByHour,
    messagesByDay,
    topWords,
    responseTime: assistantMessages.length > 0 ? 1200 : 0,
    conversationRate: userMessages.length > 0 ? assistantMessages.length / userMessages.length : 0,
  }
}

function formatNumber(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return n.toString()
}

const HOUR_LABELS = Array.from({ length: 24 }, (_, i) => `${i}`)
const DAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

export const ChatAnalyticsDashboard = memo(function ChatAnalyticsDashboard({
  messages,
  className,
}: ChatAnalyticsDashboardProps) {
  const [timeRange, setTimeRange] = useState<'all' | '24h' | '7d'>('all')

  const filteredMessages = useMemo(() => {
    if (timeRange === 'all') return messages
    const now = Date.now()
    const cutoff = timeRange === '24h' ? now - 86400000 : now - 604800000
    return messages.filter(m => m.timestamp >= cutoff)
  }, [messages, timeRange])

  const analytics = useMemo(() => calculateAnalytics(filteredMessages), [filteredMessages])

  const maxHourly = Math.max(...analytics.messagesByHour, 1)
  const maxDaily = Math.max(...analytics.messagesByDay, 1)

  if (messages.length === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-4', className)}>
        No messages to analyze
      </div>
    )
  }

  return (
    <div className={cn('space-y-4', className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium">Analytics Dashboard</span>
        <div className="flex gap-1">
          {(['all', '24h', '7d'] as const).map(range => (
            <button
              key={range}
              type="button"
              onClick={() => setTimeRange(range)}
              className={cn(
                'text-[10px] px-2 py-0.5 rounded transition-colors',
                timeRange === range ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
              )}
            >
              {range === 'all' ? 'All' : range === '24h' ? '24h' : '7d'}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <StatCard label="Messages" value={formatNumber(analytics.totalMessages)} />
        <StatCard label="Words" value={formatNumber(analytics.totalWords)} />
        <StatCard label="Characters" value={formatNumber(analytics.totalCharacters)} />
      </div>

      <div className="grid grid-cols-2 gap-2">
        <StatCard label="Avg Length" value={`${analytics.avgMessageLength} chars`} />
        <StatCard label="Avg Words" value={`${analytics.avgWordsPerMessage} words`} />
        <StatCard label="Longest" value={formatNumber(analytics.longestMessage)} />
        <StatCard label="Shortest" value={formatNumber(analytics.shortestMessage)} />
      </div>

      <div>
        <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Messages by Hour
        </h4>
        <div className="flex items-end gap-px h-16">
          {analytics.messagesByHour.map((count, i) => (
            <div
              key={i}
              className="flex-1 bg-primary/50 rounded-t"
              style={{ height: `${(count / maxHourly) * 100}%` }}
              title={`${HOUR_LABELS[i]}:00 - ${count} messages`}
            />
          ))}
        </div>
        <div className="flex justify-between text-[8px] text-muted-foreground mt-1">
          <span>0</span>
          <span>6</span>
          <span>12</span>
          <span>18</span>
          <span>23</span>
        </div>
      </div>

      <div>
        <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
          Messages by Day
        </h4>
        <div className="flex items-end gap-1 h-16">
          {analytics.messagesByDay.map((count, i) => (
            <div key={i} className="flex-1 flex flex-col items-center gap-1">
              <div
                className="w-full bg-primary/50 rounded-t"
                style={{ height: `${(count / maxDaily) * 100}%` }}
                title={`${DAY_LABELS[i]} - ${count} messages`}
              />
              <span className="text-[8px] text-muted-foreground">{DAY_LABELS[i]}</span>
            </div>
          ))}
        </div>
      </div>

      {analytics.topWords.length > 0 && (
        <div>
          <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Top Words
          </h4>
          <div className="space-y-1">
            {analytics.topWords.slice(0, 5).map(({ word, count }) => (
              <div key={word} className="flex items-center gap-2">
                <span className="text-xs flex-1 truncate">{word}</span>
                <div className="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary/50 rounded-full"
                    style={{ width: `${(count / analytics.topWords[0].count) * 100}%` }}
                  />
                </div>
                <span className="text-[10px] text-muted-foreground w-8 text-right">{count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        <StatCard label="User Messages" value={analytics.userMessages.toString()} />
        <StatCard label="AI Messages" value={analytics.assistantMessages.toString()} />
      </div>
    </div>
  )
})

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border bg-muted/30 p-2">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  )
}