'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconChat, IconCopy, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ConversationShareProps {
  messages: ChatMessage[]
  sessionName?: string
  sessionId?: string
  className?: string
}

type ShareMethod = 'link' | 'clipboard'

export const ConversationShare = memo(function ConversationShare({
  messages,
  sessionName = 'Chat',
  sessionId,
  className,
}: ConversationShareProps) {
  const [copied, setCopied] = useState<ShareMethod | null>(null)

  const formatAsMarkdown = useCallback(() => {
    const lines: string[] = []
    lines.push(`# ${sessionName}`)
    lines.push('')
    for (const msg of messages) {
      const role = msg.role === 'user' ? 'User' : 'Assistant'
      lines.push(`**${role}:** ${msg.content}`)
      lines.push('')
    }
    return lines.join('\n')
  }, [messages, sessionName])

  const handleCopyToClipboard = useCallback(async () => {
    const content = formatAsMarkdown()
    await navigator.clipboard.writeText(content)
    setCopied('clipboard')
    setTimeout(() => setCopied(null), 2000)
  }, [formatAsMarkdown])

  const handleShareLink = useCallback(async () => {
    if (sessionId) {
      const url = `${window.location.origin}/chat?session=${sessionId}`
      await navigator.clipboard.writeText(url)
      setCopied('link')
      setTimeout(() => setCopied(null), 2000)
    }
  }, [sessionId])

  if (messages.length === 0) {
    return (
      <div className={cn('text-xs text-muted-foreground text-center py-2', className)}>
        No messages to share
      </div>
    )
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">
          {messages.length} messages
        </span>
        <div className="flex items-center gap-1">
          {sessionId && (
            <Button
              variant="ghost"
              size="sm"
              className="h-6 text-xs gap-1"
              onClick={handleShareLink}
            >
              <IconChat className="h-3 w-3" />
              {copied === 'link' ? 'Copied!' : 'Link'}
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-xs gap-1"
            onClick={handleCopyToClipboard}
          >
            {copied === 'clipboard' ? <IconCheck className="h-3 w-3" /> : <IconCopy className="h-3 w-3" />}
            {copied === 'clipboard' ? 'Copied!' : 'Copy'}
          </Button>
        </div>
      </div>
    </div>
  )
})