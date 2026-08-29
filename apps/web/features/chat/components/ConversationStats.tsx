'use client'

import { useMemo, memo } from 'react'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ConversationStatsProps {
  messages: ChatMessage[]
  className?: string
}

interface Stats {
  totalMessages: number
  userMessages: number
  assistantMessages: number
  totalCharacters: number
  avgMessageLength: number
  longestMessage: number
  wordCount: number
  toolCalls: number
  images: number
}

function computeStats(messages: ChatMessage[]): Stats {
  let totalCharacters = 0
  let longestMessage = 0
  let wordCount = 0
  let toolCalls = 0
  let images = 0

  for (const msg of messages) {
    const len = msg.content.length
    totalCharacters += len
    longestMessage = Math.max(longestMessage, len)
    wordCount += msg.content.split(/\s+/).filter(Boolean).length

    if (msg.toolCalls) {
      toolCalls += msg.toolCalls.length
    }
    if (msg.images) {
      images += msg.images.length
    }
  }

  const userMessages = messages.filter(m => m.role === 'user').length
  const assistantMessages = messages.filter(m => m.role === 'assistant').length

  return {
    totalMessages: messages.length,
    userMessages,
    assistantMessages,
    totalCharacters,
    avgMessageLength: messages.length > 0 ? Math.round(totalCharacters / messages.length) : 0,
    longestMessage,
    wordCount,
    toolCalls,
    images,
  }
}

function StatItem({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="space-y-0.5">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">{label}</div>
      <div className="text-sm font-medium">{typeof value === 'number' ? value.toLocaleString() : value}</div>
    </div>
  )
}

export const ConversationStats = memo(function ConversationStats({
  messages,
  className,
}: ConversationStatsProps) {
  const stats = useMemo(() => computeStats(messages), [messages])

  if (messages.length === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-2', className)}>
        No messages yet
      </div>
    )
  }

  return (
    <div className={cn('space-y-3', className)}>
      <div className="grid grid-cols-2 gap-3">
        <StatItem label="Messages" value={stats.totalMessages} />
        <StatItem label="Words" value={stats.wordCount} />
        <StatItem label="Characters" value={stats.totalCharacters} />
        <StatItem label="Avg Length" value={`${stats.avgMessageLength} chars`} />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <StatItem label="User" value={stats.userMessages} />
        <StatItem label="Assistant" value={stats.assistantMessages} />
        {stats.toolCalls > 0 && <StatItem label="Tool Calls" value={stats.toolCalls} />}
        {stats.images > 0 && <StatItem label="Images" value={stats.images} />}
      </div>
    </div>
  )
})