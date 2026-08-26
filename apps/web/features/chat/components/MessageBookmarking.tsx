'use client'

import { useState, useCallback, useMemo, useEffect, memo } from 'react'
import { Button, IconX, IconPlus, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { ChatMessage } from '@/lib/chat-utils'

interface Bookmark {
  id: string
  messageId: string
  content: string
  category: string
  note: string
  createdAt: number
}

interface MessageBookmarkingProps {
  messages: ChatMessage[]
  onJumpToMessage: (messageId: string) => void
  className?: string
}

const DEFAULT_CATEGORIES = ['Important', 'Code', 'Question', 'Reference', 'Todo']

const STORAGE_KEY = 'message-bookmarks'

function loadBookmarks(): Bookmark[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveBookmarks(bookmarks: Bookmark[]) {
  if (typeof window === 'undefined') return
  localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks))
}

export const MessageBookmarking = memo(function MessageBookmarking({
  messages,
  onJumpToMessage,
  className,
}: MessageBookmarkingProps) {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([])
  const [filter, setFilter] = useState<string>('all')
  const [customCategory, setCustomCategory] = useState('')
  const [showAddCategory, setShowAddCategory] = useState(false)
  const [categories, setCategories] = useState<string[]>(DEFAULT_CATEGORIES)

  useEffect(() => {
    setBookmarks(loadBookmarks())
  }, [])

  const filteredBookmarks = useMemo(() => {
    if (filter === 'all') return bookmarks
    return bookmarks.filter(b => b.category === filter)
  }, [bookmarks, filter])

  const stats = useMemo(() => {
    const byCategory: Record<string, number> = {}
    for (const b of bookmarks) {
      byCategory[b.category] = (byCategory[b.category] || 0) + 1
    }
    return { total: bookmarks.length, byCategory }
  }, [bookmarks])

  const handleAddCategory = useCallback(() => {
    const trimmed = customCategory.trim()
    if (trimmed && !categories.includes(trimmed)) {
      setCategories(prev => [...prev, trimmed])
      setCustomCategory('')
      setShowAddCategory(false)
    }
  }, [customCategory, categories])

  const handleRemoveBookmark = useCallback((id: string) => {
    const next = bookmarks.filter(b => b.id !== id)
    setBookmarks(next)
    saveBookmarks(next)
  }, [bookmarks])

  const handleClearCategory = useCallback((category: string) => {
    const next = bookmarks.filter(b => b.category !== category)
    setBookmarks(next)
    saveBookmarks(next)
    setCategories(prev => prev.filter(c => c !== category))
  }, [bookmarks])

  return (
    <div className={cn('border rounded-lg bg-card overflow-hidden', className)}>
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium">Bookmarks</span>
          <span className="text-[10px] text-muted-foreground">({stats.total})</span>
        </div>
        <Button
          variant="ghost"
          size="icon-sm"
          className="h-5 w-5"
          onClick={() => setShowAddCategory(!showAddCategory)}
          aria-label="Add category"
        >
          <IconPlus className="h-3 w-3" />
        </Button>
      </div>

      {showAddCategory && (
        <div className="p-2 border-b flex gap-1">
          <input
            type="text"
            value={customCategory}
            onChange={(e) => setCustomCategory(e.target.value)}
            placeholder="Category name..."
            className="flex-1 text-xs bg-transparent border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary/50"
            onKeyDown={(e) => e.key === 'Enter' && handleAddCategory()}
          />
          <Button
            variant="ghost"
            size="icon-sm"
            className="h-6 w-6"
            onClick={handleAddCategory}
            disabled={!customCategory.trim()}
            aria-label="Save category"
          >
            <IconCheck className="h-3 w-3" />
          </Button>
        </div>
      )}

      <div className="flex flex-wrap gap-1 p-2 border-b">
        <button
          type="button"
          onClick={() => setFilter('all')}
          className={cn(
            'text-[10px] px-2 py-0.5 rounded transition-colors',
            filter === 'all' ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
          )}
        >
          All ({stats.total})
        </button>
        {categories.map(cat => (
          <button
            key={cat}
            type="button"
            onClick={() => setFilter(cat)}
            className={cn(
              'text-[10px] px-2 py-0.5 rounded transition-colors',
              filter === cat ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/50',
            )}
          >
            {cat} ({stats.byCategory[cat] || 0})
          </button>
        ))}
      </div>

      <div className="max-h-[400px] overflow-y-auto">
        {filteredBookmarks.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            {bookmarks.length === 0 ? 'No bookmarks yet' : 'No bookmarks in this category'}
          </p>
        ) : (
          <div className="divide-y">
            {filteredBookmarks.map(bookmark => (
              <div key={bookmark.id} className="px-3 py-2 hover:bg-muted/30 group">
                <div className="flex items-start justify-between gap-2">
                  <button
                    type="button"
                    className="flex-1 text-left min-w-0"
                    onClick={() => onJumpToMessage(bookmark.messageId)}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                        {bookmark.category}
                      </span>
                      <span className="text-[10px] text-muted-foreground">
                        {new Date(bookmark.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-xs truncate">{bookmark.content.slice(0, 80)}</p>
                    {bookmark.note && (
                      <p className="text-[10px] text-muted-foreground mt-1">{bookmark.note}</p>
                    )}
                  </button>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    className="h-5 w-5 opacity-0 group-hover:opacity-100 shrink-0"
                    onClick={() => handleRemoveBookmark(bookmark.id)}
                    title="Remove bookmark"
                  >
                    <IconX className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})