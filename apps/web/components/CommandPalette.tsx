'use client'

import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { cn } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { sessionController } from '@/lib/session-controller'

interface CommandAction {
  id: string
  label: string
  description: string
  icon: string
  category: 'navigation' | 'action' | 'conversation' | 'model'
  run: () => void
}

export function CommandPalette() {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [recentSessions, setRecentSessions] = useState<{ id: string; name: string }[]>([])
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    sessionController.list().then(sessions => {
      setRecentSessions(sessions.slice(0, 5).map(s => ({ id: s.id, name: s.name || 'Untitled' })))
    }).catch(() => {})
  }, [])

  const actions: CommandAction[] = useMemo(() => {
    const acts: CommandAction[] = [
      { id: 'act-newchat', label: 'New Chat', description: 'Start a new conversation', icon: '➕', category: 'action', run: () => { window.dispatchEvent(new CustomEvent('new-chat')); router.push('/chat') } },
      { id: 'act-search', label: 'Search Conversations', description: 'Search across all conversations', icon: '🔍', category: 'action', run: () => { setOpen(false); window.dispatchEvent(new CustomEvent('search-conversations')) } },
      { id: 'act-export', label: 'Export Chat', description: 'Download current chat as markdown', icon: '📥', category: 'action', run: () => { window.dispatchEvent(new CustomEvent('export-chat')); setOpen(false) } },
    ]
    const conv: CommandAction[] = recentSessions.map(s => ({
      id: `conv-${s.id}`, label: s.name, description: 'Open conversation', icon: '💭', category: 'conversation' as const, run: () => router.push(`/chat?session=${s.id}`),
    }))
    return [...acts, ...conv]
  }, [router, recentSessions])

  const filtered = useMemo(() => {
    if (!query.trim()) return actions
    const q = query.toLowerCase()
    return actions.filter(a =>
      a.label.toLowerCase().includes(q) ||
      a.description.toLowerCase().includes(q)
    )
  }, [actions, query])

  useEffect(() => {
    if (!open) { setQuery(''); setSelectedIdx(0) }
  }, [open])

  useEffect(() => {
    setSelectedIdx(0)
  }, [query])

  useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 50)
  }, [open])

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setOpen(o => !o)
      }
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('keydown', down)
    return () => window.removeEventListener('keydown', down)
  }, [])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelectedIdx(i => Math.min(i + 1, filtered.length - 1)) }
    if (e.key === 'ArrowUp') { e.preventDefault(); setSelectedIdx(i => Math.max(i - 1, 0)) }
    if (e.key === 'Enter' && filtered[selectedIdx]) {
      setOpen(false)
      filtered[selectedIdx].run()
    }
  }, [filtered, selectedIdx])

  if (!open) return null

  return (
    <>
      <div className="fixed inset-0 z-[100] bg-background/60 backdrop-blur-sm" onClick={() => setOpen(false)} />
      <div className="fixed left-1/2 top-[15%] z-[101] w-full max-w-lg -translate-x-1/2">
        <div className="overflow-hidden rounded-xl border border-border/50 bg-popover shadow-2xl">
          <div className="flex items-center border-b border-border/30 px-3">
            <span className="text-sm text-muted-foreground mr-2">🔍</span>
            <input
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search conversations, models, pages..."
              className="flex-1 bg-transparent py-3 text-sm outline-none placeholder:text-muted-foreground/50"
            />
            <kbd className="rounded border border-border/40 bg-muted/40 px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">Esc</kbd>
          </div>
          <div className="max-h-80 overflow-y-auto py-1">
            {filtered.length === 0 ? (
              <p className="px-4 py-6 text-center text-sm text-muted-foreground">No results for &ldquo;{query}&rdquo;</p>
            ) : (
              filtered.map((action, i) => {
                const cat = action.category
                const showHeader = i === 0 || filtered[i - 1].category !== cat
                const catLabel = { navigation: 'Pages', action: 'Actions', conversation: 'Conversations', model: 'Models' }[cat]
                return (
                  <div key={action.id}>
                    {showHeader && (
                      <p className="px-3 pt-2 pb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
                        {catLabel}
                      </p>
                    )}
                    <button
                      className={cn(
                        'flex w-full items-center gap-3 px-3 py-2 text-left text-sm transition-colors',
                        i === selectedIdx ? 'bg-primary/10 text-primary' : 'hover:bg-muted'
                      )}
                      onClick={() => { setOpen(false); action.run() }}
                      onMouseEnter={() => setSelectedIdx(i)}
                    >
                      <span className="text-base">{action.icon}</span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium">{action.label}</p>
                        <p className="truncate text-xs text-muted-foreground">{action.description}</p>
                      </div>
                    </button>
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>
    </>
  )
}
