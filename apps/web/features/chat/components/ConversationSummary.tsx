'use client'

import { useState, memo } from 'react'
import { Button, IconCopy, IconCheck, IconRefresh } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import { useChatSummary } from '@/features/chat/hooks/useChatSummary'
import type { ChatMessage } from '@/lib/chat-utils'

interface ConversationSummaryProps {
  messages: ChatMessage[]
  className?: string
}

export const ConversationSummary = memo(function ConversationSummary({
  messages,
  className,
}: ConversationSummaryProps) {
  const { summary, isGenerating, error, generateSummary, clearSummary } = useChatSummary({
    temperature: 0.3,
  })
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    if (summary) {
      await navigator.clipboard.writeText(summary)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleGenerate = () => {
    generateSummary(messages)
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-foreground/80">Conversation Summary</span>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 text-xs"
            onClick={handleGenerate}
            disabled={isGenerating || messages.length === 0}
          >
            <IconRefresh className={cn('h-3 w-3 mr-1', isGenerating && 'animate-spin')} />
            {isGenerating ? 'Generating...' : 'Generate'}
          </Button>
          {summary && (
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-6 w-6"
              onClick={handleCopy}
              aria-label={copied ? 'Copied' : 'Copy summary'}
            >
              {copied ? <IconCheck className="h-3 w-3" /> : <IconCopy className="h-3 w-3" />}
            </Button>
          )}
        </div>
      </div>

      {error && (
        <div className="text-xs text-destructive bg-destructive/10 rounded p-2">
          {error}
        </div>
      )}

      {summary && (
        <div className="text-xs text-foreground/70 bg-muted/30 rounded p-2 whitespace-pre-wrap">
          {summary}
        </div>
      )}

      {!summary && !isGenerating && !error && (
        <div className="text-xs text-muted-foreground text-center py-2">
          Click "Generate" to create a summary of this conversation
        </div>
      )}
    </div>
  )
})