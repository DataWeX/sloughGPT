'use client'

import { useMemo, memo } from 'react'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface PerformanceMetricsProps {
  messages: ChatMessage[]
  className?: string
}

interface Metrics {
  totalMessages: number
  userMessages: number
  assistantMessages: number
  avgResponseLength: number
  longestResponse: number
  shortestResponse: number
  totalChars: number
  avgCharsPerMessage: number
}

function calculateMetrics(messages: ChatMessage[]): Metrics {
  const userMessages = messages.filter(m => m.role === 'user')
  const assistantMessages = messages.filter(m => m.role === 'assistant')

  const responseLengths = assistantMessages.map(m => m.content.length)
  const totalChars = messages.reduce((sum, m) => sum + m.content.length, 0)

  return {
    totalMessages: messages.length,
    userMessages: userMessages.length,
    assistantMessages: assistantMessages.length,
    avgResponseLength: responseLengths.length > 0
      ? Math.round(responseLengths.reduce((a, b) => a + b, 0) / responseLengths.length)
      : 0,
    longestResponse: responseLengths.length > 0 ? Math.max(...responseLengths) : 0,
    shortestResponse: responseLengths.length > 0 ? Math.min(...responseLengths) : 0,
    totalChars,
    avgCharsPerMessage: messages.length > 0 ? Math.round(totalChars / messages.length) : 0,
  }
}

function formatNumber(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return n.toString()
}

export const PerformanceMetrics = memo(function PerformanceMetrics({
  messages,
  className,
}: PerformanceMetricsProps) {
  const metrics = useMemo(() => calculateMetrics(messages), [messages])

  if (messages.length === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-4', className)}>
        No messages yet
      </div>
    )
  }

  return (
    <div className={cn('space-y-3', className)}>
      <div className="grid grid-cols-2 gap-2">
        <MetricCard label="Total Messages" value={formatNumber(metrics.totalMessages)} />
        <MetricCard label="User Messages" value={formatNumber(metrics.userMessages)} />
        <MetricCard label="Assistant Messages" value={formatNumber(metrics.assistantMessages)} />
        <MetricCard label="Total Characters" value={formatNumber(metrics.totalChars)} />
      </div>

      {metrics.assistantMessages > 0 && (
        <div className="space-y-1">
          <h4 className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
            Response Stats
          </h4>
          <div className="grid grid-cols-3 gap-2">
            <MiniStat label="Avg Length" value={formatNumber(metrics.avgResponseLength)} />
            <MiniStat label="Longest" value={formatNumber(metrics.longestResponse)} />
            <MiniStat label="Shortest" value={formatNumber(metrics.shortestResponse)} />
          </div>
        </div>
      )}

      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>Avg chars/message: {formatNumber(metrics.avgCharsPerMessage)}</span>
        <span>Response ratio: {metrics.userMessages > 0 ? (metrics.assistantMessages / metrics.userMessages).toFixed(1) : '—'}</span>
      </div>
    </div>
  )
})

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border bg-muted/30 p-2">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="text-sm font-medium">{value}</div>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-center">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="text-xs font-medium">{value}</div>
    </div>
  )
}