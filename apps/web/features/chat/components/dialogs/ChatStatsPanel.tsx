'use client'

import { memo, useMemo } from 'react'
import { cn } from '@sloughgpt/strui'
import { IconX, IconChart } from '@sloughgpt/strui'
import type { ChatMessage } from '../types'

interface ChatStatsPanelProps {
  open: boolean
  onClose: () => void
  messages: ChatMessage[]
}

export const ChatStatsPanel = memo(function ChatStatsPanel({
  open,
  onClose,
  messages,
}: ChatStatsPanelProps) {
  const stats = useMemo(() => {
    const userMessages = messages.filter(m => m.role === 'user')
    const assistantMessages = messages.filter(m => m.role === 'assistant')
    const totalChars = messages.reduce((sum, m) => sum + m.content.length, 0)
    const userChars = userMessages.reduce((sum, m) => sum + m.content.length, 0)
    const assistantChars = assistantMessages.reduce((sum, m) => sum + m.content.length, 0)
    const avgUserLength = userMessages.length > 0 ? Math.round(userChars / userMessages.length) : 0
    const avgAssistantLength = assistantMessages.length > 0 ? Math.round(assistantChars / assistantMessages.length) : 0
    const estimatedTokens = Math.round(totalChars / 4)
    const pinnedMessages = messages.filter(m => m.pinned).length
    const messagesWithNotes = messages.filter(m => m.reactions && Object.keys(m.reactions).length > 0).length

    const firstTimestamp = messages[0]?.timestamp
    const lastTimestamp = messages[messages.length - 1]?.timestamp
    let duration = 'N/A'
    if (firstTimestamp && lastTimestamp) {
      const start = new Date(firstTimestamp).getTime()
      const end = new Date(lastTimestamp).getTime()
      const diffMs = end - start
      const diffMins = Math.round(diffMs / 60000)
      if (diffMins < 1) duration = '< 1 min'
      else if (diffMins < 60) duration = `${diffMins} min`
      else duration = `${Math.round(diffMins / 60)}h ${diffMins % 60}m`
    }

    return {
      totalMessages: messages.length,
      userMessages: userMessages.length,
      assistantMessages: assistantMessages.length,
      totalChars,
      userChars,
      assistantChars,
      avgUserLength,
      avgAssistantLength,
      estimatedTokens,
      pinnedMessages,
      messagesWithNotes,
      duration,
    }
  }, [messages])

  if (!open) return null

  const statRows = [
    { label: 'Total messages', value: stats.totalMessages },
    { label: 'Your messages', value: stats.userMessages },
    { label: 'Assistant messages', value: stats.assistantMessages },
    { label: 'Total characters', value: stats.totalChars.toLocaleString() },
    { label: 'Your characters', value: stats.userChars.toLocaleString() },
    { label: 'Assistant characters', value: stats.assistantChars.toLocaleString() },
    { label: 'Avg message length (you)', value: `${stats.avgUserLength} chars` },
    { label: 'Avg message length (assistant)', value: `${stats.avgAssistantLength} chars` },
    { label: 'Estimated tokens', value: `~${stats.estimatedTokens.toLocaleString()}` },
    { label: 'Pinned messages', value: stats.pinnedMessages },
    { label: 'Messages with reactions', value: stats.messagesWithNotes },
    { label: 'Conversation duration', value: stats.duration },
  ]

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={onClose}>
      <div
        className="bg-background border border-border rounded-lg shadow-xl w-[380px] max-h-[80vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div className="flex items-center gap-2">
            <IconChart className="h-4 w-4 text-primary" />
            <h2 className="text-sm font-medium">Conversation Statistics</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <IconX className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        <div className="p-4 overflow-y-auto max-h-[calc(80vh-52px)]">
          <div className="space-y-2">
            {statRows.map((row, i) => (
              <div
                key={i}
                className="flex items-center justify-between py-1.5 border-b border-border/40 last:border-0"
              >
                <span className="text-sm text-foreground/80">{row.label}</span>
                <span className="text-sm font-medium text-muted-foreground">{row.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
})
