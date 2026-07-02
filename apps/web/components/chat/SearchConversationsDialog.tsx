'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogPortal, DialogOverlay } from '@/components/ui/dialog'
import { IconSearch, IconMessage, IconX } from '@/components/ui'
import { chatDB } from '@/lib/db'
import type { ChatMessage } from '@/lib/chat-utils'
import { cn } from '@/lib/cn'

interface SearchResult {
  session: { id: string; name: string }
  matches: ChatMessage[]
}

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialQuery?: string
}

export function SearchConversationsDialog({ open, onOpenChange, initialQuery }: Props) {
  const router = useRouter()
  const [query, setQuery] = useState(initialQuery || '')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      if (initialQuery) setQuery(initialQuery)
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [open, initialQuery])

  useEffect(() => {
    if (!query.trim()) { setResults([]); return }
    setLoading(true)
    const timer = setTimeout(async () => {
      try {
        const res = await chatDB.searchAllSessions(query)
        setResults(res)
      } catch { setResults([]) }
      setLoading(false)
    }, 200)
    return () => clearTimeout(timer)
  }, [query])

  const handleSelect = useCallback((sessionId: string) => {
    onOpenChange(false)
    router.push(`/chat?session=${sessionId}`)
  }, [router, onOpenChange])

  const highlight = (text: string) => {
    if (!query.trim()) return text
    const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
    return parts.map((part, i) =>
      part.toLowerCase() === query.toLowerCase()
        ? <mark key={i} className="bg-primary/20 text-foreground rounded-sm px-0.5">{part}</mark>
        : part
    )
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent className="max-w-lg max-h-[80vh] flex flex-col">
          <DialogHeader>
            <DialogTitle>Search Conversations</DialogTitle>
          </DialogHeader>
          <div className="relative mb-3">
            <IconSearch className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/60 pointer-events-none" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search all conversations..."
              data-testid="search-input"
              className="w-full h-9 pl-8 pr-8 text-sm rounded-lg border border-input bg-background/80 focus:outline-none focus:ring-1 focus:ring-primary/40"
            />
            {query && (
              <button
                onClick={() => setQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                aria-label="Clear search"
              >
                <IconX className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <div className="flex-1 overflow-y-auto min-h-0 space-y-1">
            {loading ? (
              <div className="space-y-2 py-4">
                {[1,2,3].map(i => (
                  <div key={i} className="h-12 rounded-lg bg-muted/40 animate-pulse" />
                ))}
              </div>
            ) : query && results.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-8">No matches found</p>
            ) : !query ? (
              <p className="text-sm text-muted-foreground text-center py-8">Type to search across all conversations</p>
            ) : (
              results.map(r => (
                <button
                  key={r.session.id}
                  onClick={() => handleSelect(r.session.id)}
                  className="w-full text-left rounded-lg border border-border/40 p-2.5 hover:bg-muted/30 hover:border-border/60 transition-colors"
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    <IconMessage className="h-3 w-3 text-muted-foreground/60 shrink-0" />
                    <span className="text-xs font-medium truncate">{r.session.name}</span>
                    <span className="text-[10px] text-muted-foreground/60 ml-auto shrink-0">{r.matches.length} match{r.matches.length !== 1 ? 'es' : ''}</span>
                  </div>
                  {r.matches.slice(0, 2).map((m, i) => (
                    <p
                      key={i}
                      className={cn(
                        "text-xs leading-relaxed line-clamp-1 pl-5",
                        m.role === 'user' ? "text-foreground/80" : "text-muted-foreground/70"
                      )}
                    >
                      <span className="text-[10px] uppercase tracking-wider mr-1 opacity-60">{m.role}</span>
                      {highlight(m.content.slice(0, 120))}
                    </p>
                  ))}
                  {r.matches.length > 2 && (
                    <p className="text-[10px] text-muted-foreground/50 pl-5 mt-0.5">+{r.matches.length - 2} more</p>
                  )}
                </button>
              ))
            )}
          </div>
        </DialogContent>
      </DialogPortal>
    </Dialog>
  )
}
