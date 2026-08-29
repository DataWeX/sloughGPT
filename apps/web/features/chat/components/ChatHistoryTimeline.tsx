'use client'

import { useState, useMemo, useCallback, memo } from 'react'
import { Button, IconX, IconClock } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ChatHistoryTimelineProps {
  messages: ChatMessage[]
  onNavigate: (messageId: string) => void
  onClose: () => void
  className?: string
}

interface TimeGroup {
  label: string
  messages: ChatMessage[]
}

function groupMessagesByTime(messages: ChatMessage[]): TimeGroup[] {
  const groups: TimeGroup[] = []
  let currentGroup: TimeGroup | null = null

  for (const msg of messages) {
    const date = new Date(msg.timestamp)
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    const label = `${msg.role === 'user' ? 'You' : 'Assistant'} · ${timeStr}`

    if (!currentGroup || currentGroup.label !== label) {
      currentGroup = { label, messages: [msg] }
      groups.push(currentGroup)
    } else {
      currentGroup.messages.push(msg)
    }
  }

  return groups
}

function formatTimestamp(ts: number | Date): string {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export const ChatHistoryTimeline = memo(function ChatHistoryTimeline({
  messages,
  onNavigate,
  onClose,
  className,
}: ChatHistoryTimelineProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const groups = useMemo(() => groupMessagesByTime(messages), [messages])

  const handleSelect = useCallback((msgId: string) => {
    setSelectedId(msgId)
    onNavigate(msgId)
  }, [onNavigate])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <IconClock className="h-3 w-3 text-muted-foreground" />
          <span className="text-xs font-medium">History Timeline</span>
          <span className="text-[10px] text-muted-foreground">({messages.length} messages)</span>
        </div>
        <Button variant="ghost" size="icon-sm" className="h-5 w-5" onClick={onClose} aria-label="Close timeline">
          <IconX className="h-3 w-3" />
        </Button>
      </div>

      <div className="max-h-[400px] overflow-y-auto p-2">
        {groups.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">No messages</p>
        ) : (
          <div className="space-y-3">
            {groups.map((group, gi) => (
              <div key={gi}>
                <div className="text-[10px] text-muted-foreground mb-1 px-1">{group.label}</div>
                <div className="space-y-0.5">
                  {group.messages.map((msg) => (
                    <button
                      key={msg.id}
                      type="button"
                      onClick={() => handleSelect(msg.id)}
                      className={cn(
                        'w-full text-left px-2 py-1 rounded text-xs transition-colors',
                        'hover:bg-muted/50',
                        selectedId === msg.id && 'bg-primary/10 text-primary',
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="truncate">
                          {msg.content.slice(0, 60)}{msg.content.length > 60 ? '…' : ''}
                        </span>
                        <span className="text-[10px] text-muted-foreground shrink-0">
                          {formatTimestamp(msg.timestamp)}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})