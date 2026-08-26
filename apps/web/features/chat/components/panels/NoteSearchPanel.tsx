'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import { cn, Button } from '@sloughgpt/strui'
import { IconSearch, IconX } from '@sloughgpt/strui'
import { chatDB, type MessageNote } from '@/lib/db'
import { truncateMessage } from '@/lib/conversations-utils'

interface NoteSearchPanelProps {
  open: boolean
  onClose: () => void
  onNavigateToNote: (sessionId: string, messageId: string) => void
}

export function NoteSearchPanel({ open, onClose, onNavigateToNote }: NoteSearchPanelProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<MessageNote[]>([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 100)
    } else {
      setQuery('')
      setResults([])
    }
  }, [open])

  const handleSearch = useCallback(async (q: string) => {
    if (!q.trim()) {
      setResults([])
      return
    }

    setLoading(true)
    try {
      const allNotes = await chatDB.searchMessageNotes(q.trim())
      setResults(allNotes)
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }, [])

  const handleQueryChange = useCallback((value: string) => {
    setQuery(value)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => handleSearch(value), 300)
  }, [handleSearch])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose()
    }
  }, [onClose])

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[200] flex items-start justify-center pt-[10vh]">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-lg bg-background border border-border/50 rounded-xl shadow-2xl overflow-hidden">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border/50">
          <IconSearch className="h-4 w-4 text-muted-foreground/60" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => handleQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search notes across all conversations..."
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground/40"
            aria-label="Search notes"
          />
          {query && (
            <button
              type="button"
              onClick={() => { setQuery(''); setResults([]) }}
              className="h-5 w-5 flex items-center justify-center rounded hover:bg-muted/60 text-muted-foreground/60 hover:text-foreground"
              aria-label="Clear search"
            >
              <IconX className="h-3.5 w-3.5" />
            </button>
          )}
          <Button variant="ghost" size="sm" onClick={onClose} className="h-6 px-2 text-xs">
            Esc
          </Button>
        </div>

        <div className="max-h-[400px] overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="h-5 w-5 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
            </div>
          )}

          {!loading && query && results.length === 0 && (
            <div className="text-center py-8 px-4">
              <p className="text-sm text-muted-foreground">No notes found for &ldquo;{query}&rdquo;</p>
            </div>
          )}

          {!loading && results.length > 0 && (
            <div className="py-2">
              <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider px-4 py-1">
                {results.length} note{results.length !== 1 ? 's' : ''} found
              </p>
              {results.map((note) => (
                <button
                  key={`${note.sessionId}:${note.messageId}`}
                  type="button"
                  onClick={() => {
                    onNavigateToNote(note.sessionId, note.messageId)
                    onClose()
                  }}
                  className="w-full text-left px-4 py-2 hover:bg-muted/40 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-[10px] text-muted-foreground/50 font-mono">
                      {note.sessionId.slice(0, 8)}...
                    </span>
                    <span className="text-[10px] text-muted-foreground/40">
                      {new Date(note.updatedAt).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-xs text-foreground line-clamp-2">
                    {highlightQuery(note.content, query)}
                  </p>
                </button>
              ))}
            </div>
          )}

          {!loading && !query && (
            <div className="text-center py-8 px-4">
              <p className="text-sm text-muted-foreground/60">Type to search notes across all conversations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function highlightQuery(text: string, query: string): React.ReactNode {
  if (!query) return text
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase()
      ? <mark key={i} className="bg-primary/20 rounded px-0.5">{part}</mark>
      : part
  )
}
