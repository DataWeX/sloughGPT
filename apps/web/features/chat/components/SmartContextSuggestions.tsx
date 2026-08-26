'use client'

import { useState, useCallback, useMemo, memo } from 'react'
import { Button, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface ContextSuggestion {
  id: string
  type: 'file' | 'url' | 'concept' | 'previous'
  label: string
  value: string
  relevance: number
}

interface SmartContextSuggestionsProps {
  messages: ChatMessage[]
  currentInput: string
  onSelect: (context: string) => void
  className?: string
}

function extractConcepts(messages: ChatMessage[]): string[] {
  const concepts = new Set<string>()
  const keywords = [
    'react', 'javascript', 'typescript', 'python', 'api', 'database',
    'function', 'component', 'state', 'hook', 'props', 'error',
    'test', 'deploy', 'build', 'install', 'config', 'server',
  ]

  for (const msg of messages) {
    const lower = msg.content.toLowerCase()
    for (const kw of keywords) {
      if (lower.includes(kw)) {
        concepts.add(kw)
      }
    }
  }

  return Array.from(concepts).slice(0, 10)
}

function extractUrls(messages: ChatMessage[]): string[] {
  const urls = new Set<string>()
  const urlPattern = /https?:\/\/[^\s]+/g

  for (const msg of messages) {
    const matches = msg.content.match(urlPattern)
    if (matches) {
      for (const url of matches) {
        urls.add(url)
      }
    }
  }

  return Array.from(urls).slice(0, 5)
}

function extractFiles(messages: ChatMessage[]): string[] {
  const files = new Set<string>()
  const filePattern = /(?:^|\s)([\w\-./]+\.\w{1,4})(?:\s|$)/g

  for (const msg of messages) {
    const matches = msg.content.match(filePattern)
    if (matches) {
      for (const match of matches) {
        const file = match.trim()
        if (file.includes('/') || file.includes('.')) {
          files.add(file)
        }
      }
    }
  }

  return Array.from(files).slice(0, 5)
}

export const SmartContextSuggestions = memo(function SmartContextSuggestions({
  messages,
  currentInput,
  onSelect,
  className,
}: SmartContextSuggestionsProps) {
  const [dismissed, setDismissed] = useState(false)

  const suggestions = useMemo(() => {
    if (!currentInput || currentInput.length < 3) return []

    const results: ContextSuggestion[] = []
    const lower = currentInput.toLowerCase()

    const concepts = extractConcepts(messages)
    for (const concept of concepts) {
      if (lower.includes(concept)) {
        results.push({
          id: `concept-${concept}`,
          type: 'concept',
          label: concept,
          value: concept,
          relevance: 0.8,
        })
      }
    }

    const files = extractFiles(messages)
    for (const file of files) {
      if (lower.includes(file.split('/').pop()?.split('.')[0] || '')) {
        results.push({
          id: `file-${file}`,
          type: 'file',
          label: file,
          value: file,
          relevance: 0.9,
        })
      }
    }

    const urls = extractUrls(messages)
    for (const url of urls) {
      const domain = new URL(url).hostname
      if (lower.includes(domain)) {
        results.push({
          id: `url-${url}`,
          type: 'url',
          label: domain,
          value: url,
          relevance: 0.7,
        })
      }
    }

    const recentMessages = messages.slice(-5)
    for (const msg of recentMessages) {
      const words = msg.content.split(/\s+/).slice(0, 3).join(' ')
      if (words && lower.includes(words.toLowerCase().slice(0, 5))) {
        results.push({
          id: `prev-${msg.id}`,
          type: 'previous',
          label: words,
          value: msg.content.slice(0, 100),
          relevance: 0.6,
        })
      }
    }

    return results
      .sort((a, b) => b.relevance - a.relevance)
      .slice(0, 5)
  }, [messages, currentInput])

  const handleDismiss = useCallback(() => {
    setDismissed(true)
  }, [])

  if (dismissed || suggestions.length === 0) return null

  return (
    <div className={cn('flex flex-wrap items-center gap-1', className)}>
      <span className="text-[10px] text-muted-foreground">Context:</span>
      {suggestions.map(suggestion => (
        <Button
          key={suggestion.id}
          variant="ghost"
          size="sm"
          className="text-[10px] h-5 rounded-full border border-border/50"
          onClick={() => onSelect(suggestion.value)}
        >
          <span className="text-[10px] text-muted-foreground mr-1">
            {suggestion.type === 'file' ? '📄' :
             suggestion.type === 'url' ? '🔗' :
             suggestion.type === 'concept' ? '💡' : '💬'}
          </span>
          {suggestion.label}
        </Button>
      ))}
      <Button
        variant="ghost"
        size="icon-sm"
        className="h-4 w-4"
        onClick={handleDismiss}
        title="Dismiss"
      >
        <IconX className="h-2.5 w-2.5" />
      </Button>
    </div>
  )
})