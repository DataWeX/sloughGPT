'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Input } from '@sloughgpt/strui'
import { IconSearch, IconX, IconMessage } from '@sloughgpt/strui'
import { chatDB, type ChatSession, type ChatMessage as DBChatMessage } from '@/lib/db'
import { sessionController, type SearchResult as RemoteSearchResult } from '@/lib/session-controller'

interface SearchResult {
  session: ChatSession
  matches: DBChatMessage[]
  remote?: boolean
}

interface ConversationSearchProps {
  open: boolean
  onClose: () => void
  onNavigate: (sessionId: string) => void
}

function truncate(text: string, len = 80): string {
  return text.length > len ? text.slice(0, len) + '…' : text
}

function snippet(content: string, query: string, maxLen = 100): React.ReactNode {
  const q = query.toLowerCase()
  const idx = content.toLowerCase().indexOf(q)
  if (idx === -1) return truncate(content, maxLen)
  const start = Math.max(0, idx - 40)
  const end = Math.min(content.length, idx + q.length + 40)
  let s = content.slice(start, end)
  if (start > 0) s = `…${s}`
  if (end < content.length) s = `${s}…`
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = s.split(new RegExp(`(${escaped})`, 'gi'))
  return parts.map((part, i) =>
    part.toLowerCase() === q ? <mark key={i} className="bg-primary/20 rounded px-0.5">{part}</mark> : part
  )
}

export function ConversationSearch({ open, onClose, onNavigate }: ConversationSearchProps) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  function remoteMatchToLocal(rm: RemoteSearchResult): SearchResult {
    return {
      session: {
        id: rm.id,
        name: rm.name,
        messages: rm.matches.map(m => ({
          id: `remote-${rm.id}-${m.timestamp}`,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          timestamp: new Date(m.timestamp),
        })),
        createdAt: rm.created_at,
        updatedAt: rm.updated_at,
        synced: true,
        starred: false,
        pinned: false,
      },
      matches: rm.matches.map(m => ({
        id: `remote-${rm.id}-${m.timestamp}`,
        role: m.role as 'user' | 'assistant',
        content: m.content,
        timestamp: new Date(m.timestamp),
      })),
      remote: true,
    }
  }

  const doSearch = useCallback(async (q: string) => {
    if (!q.trim()) { setResults([]); return }
    setLoading(true)
    try {
      const [local, remote] = await Promise.all([
        chatDB.searchAllSessions(q),
        sessionController.search(q, 10).catch(() => [] as RemoteSearchResult[]),
      ])

      // Merge: local takes priority, remote fills gaps
      const localIds = new Set(local.map(r => r.session.id))
      const remoteResults = remote.filter(r => !localIds.has(r.id)).map(remoteMatchToLocal)
      setResults([...local, ...remoteResults])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!open) { setQuery(''); setResults([]); return }
    inputRef.current?.focus()
  }, [open])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(query), 200)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [query, doSearch])

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh]" onClick={onClose}>
      <div className="fixed inset-0 bg-black/40" />
      <div
        className="relative z-10 w-full max-w-xl mx-4 bg-background rounded-xl border border-border shadow-2xl overflow-hidden"
        onClick={e => e.stopPropagation()}
        role="dialog"
        aria-label="Search all conversations"
      >
        <div className="flex items-center gap-2 px-3 py-2.5 border-b border-border/50">
          <IconSearch className="h-4 w-4 text-muted-foreground shrink-0" />
          <Input
            ref={inputRef}
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search all conversations…"
            className="border-0 shadow-none h-8 text-sm px-0 focus-visible:ring-0"
          />
          {query && (
            <button type="button" onClick={() => { setQuery(''); setResults([]); inputRef.current?.focus() }} className="text-muted-foreground hover:text-foreground p-0.5" aria-label="Clear">
              <IconX className="h-4 w-4" />
            </button>
          )}
          <kbd className="hidden sm:inline-flex text-[10px] text-muted-foreground/50 border border-border/40 rounded px-1.5 py-0.5 font-mono">Esc</kbd>
        </div>

        <div className="max-h-[50vh] overflow-y-auto p-2">
          {loading && (
            <div className="flex items-center justify-center py-8">
              <div className="h-4 w-4 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
            </div>
          )}

          {!loading && query && results.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-8">
              No results for &ldquo;{query}&rdquo;
            </p>
          )}

          {!loading && results.length > 0 && (
            <div className="space-y-1">
              <p className="text-[10px] text-muted-foreground/60 px-1.5 py-1 uppercase tracking-wider font-medium">
                {results.length} conversation{results.length !== 1 ? 's' : ''} found
              </p>
              {results.map(r => (
                <button
                  key={r.session.id}
                  onClick={() => { onNavigate(r.session.id); onClose() }}
                  className="w-full text-left p-2.5 rounded-lg hover:bg-muted/50 transition-colors border border-transparent hover:border-border/40"
                >
                  <div className="flex items-center gap-2 mb-1.5">
                    <IconMessage className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    <span className="text-sm font-medium truncate">{r.session.name}</span>
                    {r.remote && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium shrink-0">remote</span>
                    )}
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium shrink-0">
                      {r.matches.length} match{r.matches.length !== 1 ? 'es' : ''}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground/70 leading-relaxed pl-5.5">
                    {snippet(r.matches[0]?.content || r.session.name, query)}
                  </p>
                </button>
              ))}
            </div>
          )}

          {!query && !loading && (
            <div className="py-8 text-center">
              <IconSearch className="h-8 w-8 text-muted-foreground/20 mx-auto mb-2" />
              <p className="text-sm text-muted-foreground/50">Type to search across all conversations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
