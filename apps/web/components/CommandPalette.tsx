'use client'

import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { cn } from '@sloughgpt/strui'
import { modelController } from '@/lib/model-controller'
import { sessionController } from '@/lib/session-controller'
import { soulsController, type Soul } from '@/lib/souls-controller'
import { useSettings, useUpdateSettings } from '@/lib/store'
import { useModels, useSouls } from '@/lib/query/api-hooks'
import { useToastStore } from '@/lib/toast-store'
import { NAV_SECTIONS } from '@/lib/navigation'

interface CommandAction {
  id: string
  label: string
  description: string
  icon: string
  category: 'navigation' | 'action' | 'conversation' | 'model' | 'soul'
  run: () => void
}

export function CommandPalette() {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [recentSessions, setRecentSessions] = useState<{ id: string; name: string }[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const settings = useSettings()
  const updateSettings = useUpdateSettings()
  const addToast = useToastStore(s => s.addToast)

  // Use shared query cache for models and souls — eliminates 2 independent API calls.
  const { data: modelsData } = useModels()
  const { data: soulsData } = useSouls()

  const models = useMemo(() => {
    const list = modelsData ?? []
    return list.map((model) => ({
      id: model.id || model.name,
      name: (model.id || model.name).replace(/^hf\//, ''),
      loaded: model.loaded || false,
    }))
  }, [modelsData])

  const souls = useMemo(() => {
    const list = soulsData?.souls || []
    return list.map((soul: Soul) => ({ name: soul.name, description: soul.description }))
  }, [soulsData])

  useEffect(() => {
    let active = true
    sessionController.list().then(sessions => {
      if (active) setRecentSessions(sessions.slice(0, 5).map(s => ({ id: s.id, name: s.name || 'Untitled' })))
    }).catch(() => { if (active && !loadError) setLoadError('Some data unavailable') })
    return () => { active = false }
  }, [])

  const actions: CommandAction[] = useMemo(() => {
    const nav: CommandAction[] = NAV_SECTIONS.flatMap(section =>
      section.routes.map(route => ({
        id: `nav-${route.path}`,
        label: route.description || route.path,
        description: route.description || '',
        icon: route.icon || '📄',
        category: 'navigation' as const,
        run: () => router.push(route.path),
      }))
    )

    const modelActs: CommandAction[] = models.map(m => ({
      id: `model-${m.id}`, label: `Switch to ${m.name}`, description: m.loaded ? 'Currently loaded' : 'Load and switch',
      icon: m.loaded ? '✓' : '🧠', category: 'model' as const,
      run: async () => {
        if (!m.loaded) {
          try { await modelController.load(m.id) }
          catch { addToast('Could not load model', 'error') }
        }
        router.push('/chat')
      },
    }))

    const acts: CommandAction[] = [
      { id: 'act-newchat', label: 'New Chat', description: 'Start a new conversation', icon: '➕', category: 'action', run: () => { window.dispatchEvent(new CustomEvent('new-chat')); router.push('/chat') } },
      { id: 'act-search', label: 'Search Conversations', description: 'Search across all conversations', icon: '🔍', category: 'action', run: () => { setOpen(false); window.dispatchEvent(new CustomEvent('search-conversations')) } },
      { id: 'act-export', label: 'Export Chat', description: 'Download current chat as markdown', icon: '📥', category: 'action', run: () => { window.dispatchEvent(new CustomEvent('export-chat')); setOpen(false) } },
      { id: 'act-theme', label: `Switch to ${settings.theme === 'dark' ? 'Light' : 'Dark'} Mode`, description: 'Toggle theme', icon: '🌓', category: 'action', run: () => updateSettings({ theme: settings.theme === 'dark' ? 'light' : 'dark' }) },
      { id: 'act-clear', label: 'Clear Chat History', description: 'Remove all saved conversations', icon: '🗑️', category: 'action', run: () => { localStorage.removeItem('man_current_conversation'); window.location.reload() } },
      { id: 'act-shortcuts', label: 'Keyboard Shortcuts', description: 'View all shortcuts', icon: '⌨️', category: 'action', run: () => { window.dispatchEvent(new CustomEvent('open-shortcuts')); setOpen(false) } },
    ]

    const conv: CommandAction[] = recentSessions.map(s => ({
      id: `conv-${s.id}`, label: s.name, description: 'Open conversation', icon: '💭', category: 'conversation' as const, run: () => router.push(`/chat?session=${s.id}`),
    }))

    const soulActs: CommandAction[] = souls.map((s: { name: string; description?: string }) => ({
      id: `soul-${s.name}`, label: `Switch soul: ${s.name}`, description: s.description || 'Switch personality',
      icon: '🎭', category: 'soul' as const,
      run: async () => { await soulsController.switch(s.name); router.push('/chat') },
    }))

    return [...nav, ...modelActs, ...soulActs, ...acts, ...conv]
  }, [router, recentSessions, models, souls, settings, updateSettings])

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

  useEffect(() => { setSelectedIdx(0) }, [query])

  useEffect(() => {
    if (!open) return
    const id = setTimeout(() => inputRef.current?.focus(), 50)
    return () => clearTimeout(id)
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
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Command palette"
          className="overflow-hidden rounded-xl border border-border/50 bg-popover shadow-2xl"
          onKeyDown={(e) => {
            if (e.key === 'Tab') {
              e.preventDefault()
              const focusable = e.currentTarget.querySelectorAll('input, button:not([disabled])')
              if (focusable.length === 0) return
              const first = focusable[0] as HTMLElement
              const last = focusable[focusable.length - 1] as HTMLElement
              if (e.shiftKey) {
                if (document.activeElement === first) { last.focus(); e.preventDefault() }
              } else {
                if (document.activeElement === last) { first.focus(); e.preventDefault() }
              }
            }
          }}
        >
          <div className="flex items-center border-b border-border/30 px-3">
            <span className="text-sm text-muted-foreground mr-2">🔍</span>
            <input
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search pages, models, actions..."
              aria-label="Search pages, models, actions"
              className="flex-1 bg-transparent py-3 text-sm outline-none focus:ring-2 focus:ring-ring/30 rounded placeholder:text-muted-foreground/50"
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
                const catLabel = { navigation: 'Pages', model: 'Models', soul: 'Souls', action: 'Actions', conversation: 'Conversations' }[cat]
                return (
                  <div key={action.id}>
                    {showHeader && (
                      <p className="px-3 pt-2 pb-1 text-[10px] font-medium uppercase tracking-wider text-muted-foreground/60">
                        {catLabel}
                      </p>
                    )}
                    <button
                      type="button"
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
          {loadError && (
            <div className="border-t border-border/30 px-3 py-1.5">
              <p className="text-[10px] text-muted-foreground">{loadError}</p>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
