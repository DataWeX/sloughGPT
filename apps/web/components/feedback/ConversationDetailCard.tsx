'use client'

import { useState } from 'react'
import { cn, Card, CardHeader, CardTitle, CardContent, Button } from '@sloughgpt/strui'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}

interface ConversationDetailCardProps {
  conversationId: string
  conversationName: string
  messages: Message[]
  onClose?: () => void
}

function formatTimestamp(ts?: string): string {
  if (!ts) return ''
  try {
    return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ts
  }
}

export function ConversationDetailCard({ conversationId, conversationName, messages, onClose }: ConversationDetailCardProps) {
  const [expanded, setExpanded] = useState<Record<number, boolean>>({})

  const toggleExpand = (i: number) => {
    setExpanded(prev => ({ ...prev, [i]: !prev[i] }))
  }

  return (
    <Card data-testid="conversation-detail">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="min-w-0 flex-1">
            <CardTitle className="text-base truncate">{conversationName}</CardTitle>
            <p className="text-[10px] text-muted-foreground mt-0.5">
              {messages.length} message{messages.length !== 1 ? 's' : ''} · {conversationId.slice(0, 8)}
            </p>
          </div>
          {onClose && (
            <Button size="sm" variant="ghost" className="h-6 text-[10px]" onClick={onClose}>
              Close
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {messages.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-3">No messages in this conversation.</p>
        ) : (
          <div className="space-y-2 max-h-80 overflow-y-auto">
            {messages.map((msg, i) => {
              const isLong = msg.content.length > 200
              const displayContent = isLong && !expanded[i]
                ? msg.content.slice(0, 200) + '...'
                : msg.content

              return (
                <div
                  key={i}
                  className={cn(
                    'rounded-lg px-3 py-2 text-xs',
                    msg.role === 'user'
                      ? 'bg-primary/10 ml-4'
                      : 'bg-muted/50 mr-4'
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={cn(
                      'text-[9px] font-medium uppercase tracking-wide',
                      msg.role === 'user' ? 'text-primary' : 'text-muted-foreground'
                    )}>
                      {msg.role}
                    </span>
                    {msg.timestamp && (
                      <span className="text-[9px] text-muted-foreground/60">{formatTimestamp(msg.timestamp)}</span>
                    )}
                  </div>
                  <p className="whitespace-pre-wrap text-[11px] leading-relaxed">{displayContent}</p>
                  {isLong && (
                    <button
                      type="button"
                      className="text-[9px] text-primary hover:text-primary/80 mt-1"
                      onClick={() => toggleExpand(i)}
                    >
                      {expanded[i] ? 'Show less' : 'Show more'}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
