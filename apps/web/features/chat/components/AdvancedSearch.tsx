'use client'

import { useState, useCallback, useMemo, memo } from 'react'
import { Button, IconSearch, IconX, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface AdvancedSearchProps {
  messages: ChatMessage[]
  onHighlight: (messageId: string) => void
  className?: string
}

interface SearchOptions {
  caseSensitive: boolean
  wholeWord: boolean
  useRegex: boolean
}

interface SearchResult {
  messageId: string
  content: string
  matchStart: number
  matchEnd: number
  matchText: string
}

function searchMessages(
  messages: ChatMessage[],
  query: string,
  options: SearchOptions,
): SearchResult[] {
  if (!query) return []

  const results: SearchResult[] = []

  try {
    let regex: RegExp

    if (options.useRegex) {
      regex = new RegExp(query, options.caseSensitive ? 'g' : 'gi')
    } else {
      const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      const pattern = options.wholeWord ? `\\b${escaped}\\b` : escaped
      regex = new RegExp(pattern, options.caseSensitive ? 'g' : 'gi')
    }

    for (const msg of messages) {
      let match: RegExpExecArray | null
      while ((match = regex.exec(msg.content)) !== null) {
        results.push({
          messageId: msg.id,
          content: msg.content,
          matchStart: match.index,
          matchEnd: match.index + match[0].length,
          matchText: match[0],
        })
        if (!options.useRegex) break
      }
    }
  } catch {
    // Invalid regex
  }

  return results
}

export const AdvancedSearch = memo(function AdvancedSearch({
  messages,
  onHighlight,
  className,
}: AdvancedSearchProps) {
  const [query, setQuery] = useState('')
  const [options, setOptions] = useState<SearchOptions>({
    caseSensitive: false,
    wholeWord: false,
    useRegex: false,
  })
  const [currentMatch, setCurrentMatch] = useState(0)

  const results = useMemo(
    () => searchMessages(messages, query, options),
    [messages, query, options],
  )

  const handleNext = useCallback(() => {
    if (results.length > 0) {
      const next = (currentMatch + 1) % results.length
      setCurrentMatch(next)
      onHighlight(results[next].messageId)
    }
  }, [results, currentMatch, onHighlight])

  const handlePrev = useCallback(() => {
    if (results.length > 0) {
      const prev = currentMatch === 0 ? results.length - 1 : currentMatch - 1
      setCurrentMatch(prev)
      onHighlight(results[prev].messageId)
    }
  }, [results, currentMatch, onHighlight])

  const handleClear = useCallback(() => {
    setQuery('')
    setCurrentMatch(0)
  }, [])

  const toggleOption = useCallback((key: keyof SearchOptions) => {
    setOptions(prev => ({ ...prev, [key]: !prev[key] }))
    setCurrentMatch(0)
  }, [])

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <div className="flex items-center gap-1 flex-1">
        <IconSearch className="h-3 w-3 text-muted-foreground shrink-0" />
        <input
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setCurrentMatch(0) }}
          placeholder="Search messages..."
          className="flex-1 text-xs bg-transparent border-0 p-0 focus:outline-none focus:ring-0 placeholder:text-muted-foreground/50"
        />
        {query && (
          <span className="text-[10px] text-muted-foreground shrink-0">
            {results.length > 0 ? `${currentMatch + 1}/${results.length}` : 'No matches'}
          </span>
        )}
      </div>

      <div className="flex items-center gap-0.5 shrink-0">
        <button
          type="button"
          onClick={() => toggleOption('caseSensitive')}
          className={cn(
            'text-[10px] px-1 py-0.5 rounded',
            options.caseSensitive ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
          )}
          title="Case sensitive"
        >
          Aa
        </button>
        <button
          type="button"
          onClick={() => toggleOption('wholeWord')}
          className={cn(
            'text-[10px] px-1 py-0.5 rounded',
            options.wholeWord ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
          )}
          title="Whole word"
        >
          Ab
        </button>
        <button
          type="button"
          onClick={() => toggleOption('useRegex')}
          className={cn(
            'text-[10px] px-1 py-0.5 rounded',
            options.useRegex ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
          )}
          title="Regex"
        >
          .*
        </button>
      </div>

      {query && (
        <div className="flex items-center gap-0.5 shrink-0">
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5"
            onClick={handlePrev}
            disabled={results.length === 0}
            aria-label="Previous match"
          >
            <span className="text-xs">↑</span>
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5"
            onClick={handleNext}
            disabled={results.length === 0}
            aria-label="Next match"
          >
            <span className="text-xs">↓</span>
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-5 w-5"
            onClick={handleClear}
            aria-label="Clear search"
          >
            <IconX className="h-3 w-3" />
          </Button>
        </div>
      )}
    </div>
  )
})