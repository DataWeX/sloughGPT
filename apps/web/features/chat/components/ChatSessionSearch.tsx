'use client'

import { useState, useCallback, useMemo, useEffect, memo } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

type SortOption = 'updated' | 'created' | 'title' | 'messages'
type SortDirection = 'asc' | 'desc'

interface Session {
  id: string
  title: string
  messageCount: number
  createdAt: number
  updatedAt: number
}

interface ChatSessionSearchProps {
  sessions: Session[]
  onFiltered: (filtered: Session[]) => void
  className?: string
}

const SORT_OPTIONS: { value: SortOption; label: string }[] = [
  { value: 'updated', label: 'Last Updated' },
  { value: 'created', label: 'Date Created' },
  { value: 'title', label: 'Title' },
  { value: 'messages', label: 'Messages' },
]

export const ChatSessionSearch = memo(function ChatSessionSearch({
  sessions,
  onFiltered,
  className,
}: ChatSessionSearchProps) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortOption>('updated')
  const [direction, setDirection] = useState<SortDirection>('desc')
  const [activeTab, setActiveTab] = useState<'search' | 'sort'>('search')

  const filtered = useMemo(() => {
    let result = sessions

    if (query.trim()) {
      const lower = query.toLowerCase()
      result = result.filter(s =>
        s.title.toLowerCase().includes(lower)
      )
    }

    result = [...result].sort((a, b) => {
      let cmp = 0
      switch (sort) {
        case 'updated':
          cmp = a.updatedAt - b.updatedAt
          break
        case 'created':
          cmp = a.createdAt - b.createdAt
          break
        case 'title':
          cmp = a.title.localeCompare(b.title)
          break
        case 'messages':
          cmp = a.messageCount - b.messageCount
          break
      }
      return direction === 'desc' ? -cmp : cmp
    })

    return result
  }, [sessions, query, sort, direction])

  useEffect(() => {
    onFiltered(filtered)
  }, [filtered, onFiltered])

  const handleSort = useCallback((option: SortOption) => {
    if (option === sort) {
      setDirection(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSort(option)
      setDirection('desc')
    }
  }, [sort])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center border-b bg-muted/30">
        <button
          type="button"
          onClick={() => setActiveTab('search')}
          className={cn(
            'flex-1 text-xs py-2 text-center transition-colors',
            activeTab === 'search'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          Search
        </button>
        <button
          type="button"
          onClick={() => setActiveTab('sort')}
          className={cn(
            'flex-1 text-xs py-2 text-center transition-colors',
            activeTab === 'sort'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          Sort
        </button>
      </div>

      {activeTab === 'search' ? (
        <div className="p-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search sessions..."
            className="w-full text-xs bg-transparent border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
          {query && (
            <div className="mt-2 text-[10px] text-muted-foreground">
              {filtered.length} of {sessions.length} sessions
            </div>
          )}
        </div>
      ) : (
        <div className="p-2 space-y-1">
          {SORT_OPTIONS.map(option => (
            <button
              key={option.value}
              type="button"
              onClick={() => handleSort(option.value)}
              className={cn(
                'w-full text-left text-xs px-2 py-1.5 rounded flex items-center justify-between transition-colors',
                sort === option.value
                  ? 'bg-primary/10 text-primary'
                  : 'hover:bg-muted/50 text-muted-foreground',
              )}
            >
              <span>{option.label}</span>
              {sort === option.value && (
                <span className="text-[10px]">
                  {direction === 'desc' ? '↓' : '↑'}
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      {query && filtered.length === 0 && (
        <div className="px-3 py-4 text-center text-xs text-muted-foreground">
          No sessions match "{query}"
        </div>
      )}
    </div>
  )
})