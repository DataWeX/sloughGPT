'use client'

import { useState, useCallback, useMemo, memo, useEffect } from 'react'
import { Button, IconX, IconRefresh } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface SmartReplySuggestionsProps {
  messages: ChatMessage[]
  onSelect: (suggestion: string) => void
  className?: string
}

interface SuggestionPattern {
  trigger: RegExp
  suggestions: string[]
}

const patterns: SuggestionPattern[] = [
  {
    trigger: /what|how|explain|tell me about/i,
    suggestions: [
      'Can you explain that in more detail?',
      'What are the key takeaways?',
      'Can you give an example?',
      'How does this compare to alternatives?',
    ],
  },
  {
    trigger: /help|assist|support|problem|issue|error/i,
    suggestions: [
      'Can you help me debug this?',
      'What am I missing?',
      'Is there a simpler approach?',
      'Can you walk me through the steps?',
    ],
  },
  {
    trigger: /code|function|implement|build|create|write/i,
    suggestions: [
      'Can you show me the code?',
      'What about error handling?',
      'Is this the best practice?',
      'Can you optimize this?',
    ],
  },
  {
    trigger: /think|opinion|prefer|recommend|suggest/i,
    suggestions: [
      'What do you think?',
      'Do you agree?',
      'Any other options?',
      'Which would you choose?',
    ],
  },
  {
    trigger: /done|finished|complete|success|great|thanks/i,
    suggestions: [
      "What's next?",
      'Is there anything else?',
      'Can you summarize what we did?',
      'Thanks, that was helpful!',
    ],
  },
]

const defaultSuggestions = [
  'Tell me more',
  'Can you explain?',
  'What else?',
  'Good point',
]

function generateSuggestions(lastMessage: string): string[] {
  if (!lastMessage) return defaultSuggestions

  for (const pattern of patterns) {
    if (pattern.trigger.test(lastMessage)) {
      return pattern.suggestions
    }
  }

  return defaultSuggestions
}

export const SmartReplySuggestions = memo(function SmartReplySuggestions({
  messages,
  onSelect,
  className,
}: SmartReplySuggestionsProps) {
  const [dismissed, setDismissed] = useState(false)
  const [customCount, setCustomCount] = useState(0)

  const lastAssistantMessage = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'assistant') {
        return messages[i].content
      }
    }
    return ''
  }, [messages])

  const suggestions = useMemo(
    () => generateSuggestions(lastAssistantMessage),
    [lastAssistantMessage, customCount],
  )

  useEffect(() => {
    setDismissed(false)
    setCustomCount(0)
  }, [lastAssistantMessage])

  const handleRefresh = useCallback(() => {
    setCustomCount(prev => prev + 1)
  }, [])

  const handleDismiss = useCallback(() => {
    setDismissed(true)
  }, [])

  if (dismissed || !lastAssistantMessage) return null

  return (
    <div className={cn('flex flex-wrap items-center gap-1', className)}>
      {suggestions.map((suggestion, i) => (
        <Button
          key={`${suggestion}-${i}`}
          variant="ghost"
          size="sm"
          className="text-[10px] h-6 rounded-full border border-border/50 hover:bg-muted/50"
          onClick={() => onSelect(suggestion)}
        >
          {suggestion}
        </Button>
      ))}
      <Button
        variant="ghost"
        size="icon-sm"
        className="h-5 w-5"
        onClick={handleRefresh}
        title="Refresh suggestions"
      >
        <IconRefresh className="h-3 w-3" />
      </Button>
      <Button
        variant="ghost"
        size="icon-sm"
        className="h-5 w-5"
        onClick={handleDismiss}
        title="Dismiss"
      >
        <IconX className="h-3 w-3" />
      </Button>
    </div>
  )
})