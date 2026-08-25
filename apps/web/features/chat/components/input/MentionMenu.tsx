'use client'

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { cn } from '@sloughgpt/strui'
import { agentsController, modelController } from '@/lib/controllers'
import { logger } from '@/lib/dev-log'

interface MentionItem {
  id: string
  name: string
  type: 'agent' | 'model'
  description?: string
}

interface MentionMenuProps {
  value: string
  onInsert: (text: string) => void
  onClose: () => void
}

function fuzzyScore(query: string, target: string): number {
  const q = query.toLowerCase()
  const t = target.toLowerCase()
  let qi = 0
  let score = 0
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += 1 + (qi === 0 ? 5 : 0)
      qi++
    }
  }
  return qi === q.length ? score : -1
}

export function MentionMenu({ value, onInsert, onClose }: MentionMenuProps) {
  const [items, setItems] = useState<MentionItem[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    Promise.all([
      agentsController.list().catch((e) => { logger.debug('Could not load agents for mention menu', { e }); return [] }),
      modelController.list().catch((e) => { logger.debug('Could not load models for mention menu', { e }); return [] }),
    ]).then(([agentsRes, models]) => {
      if (cancelled) return
      const agentItems: MentionItem[] = (Array.isArray(agentsRes) ? agentsRes : []).map((a: { id: string; name: string; description?: string }) => ({
        id: a.id,
        name: a.name,
        type: 'agent' as const,
        description: a.description,
      }))
      const modelItems: MentionItem[] = (models || []).map((m: { id: string; name: string; description?: string }) => ({
        id: m.id,
        name: m.name,
        type: 'model' as const,
        description: m.description,
      }))
      setItems([...agentItems, ...modelItems])
      setLoading(false)
    })
    return () => { cancelled = true }
  }, [])

  const query = value.startsWith('@') ? value.slice(1) : ''

  const scored = useMemo(() => {
    if (!query) return items.map(item => ({ item, score: 0 }))
    return items
      .map(item => {
        const nameScore = fuzzyScore(query, item.name)
        const descScore = item.description ? fuzzyScore(query, item.description) : -1
        const score = Math.max(nameScore, descScore)
        return { item, score }
      })
      .filter(x => x.score >= 0)
      .sort((a, b) => b.score - a.score)
  }, [query, items])

  const [selectedIndex, setSelectedIndex] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  useEffect(() => {
    itemRefs.current = itemRefs.current.slice(0, scored.length)
  }, [scored.length])

  useEffect(() => {
    const el = itemRefs.current[selectedIndex]
    if (el && typeof el.scrollIntoView === 'function') {
      el.scrollIntoView({ block: 'nearest' })
    }
  }, [selectedIndex])

  const select = useCallback((item: MentionItem) => {
    onInsert(`@${item.name} `)
    onClose()
  }, [onInsert, onClose])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault()
        setSelectedIndex(i => (i + 1) % scored.length)
        break
      case 'ArrowUp':
        e.preventDefault()
        setSelectedIndex(i => (i - 1 + scored.length) % scored.length)
        break
      case 'Enter':
      case 'Tab':
        e.preventDefault()
        if (scored[selectedIndex]) select(scored[selectedIndex].item)
        break
      case 'Escape':
        e.preventDefault()
        onClose()
        break
    }
  }, [scored, selectedIndex, select, onClose])

  if (loading) return null
  if (scored.length === 0) return null

  return (
    <div
      ref={listRef}
      role="listbox"
      aria-label="Mention an agent or model"
      className="absolute bottom-full left-0 right-0 mb-1 mx-1 max-h-64 overflow-y-auto rounded-lg border border-border/40 bg-popover/95 backdrop-blur-sm shadow-xl z-50"
      onKeyDown={handleKeyDown}
    >
      {scored.map(({ item, score }, i) => (
        <button
          key={`${item.type}-${item.id}`}
          ref={el => { itemRefs.current[i] = el }}
          type="button"
          role="option"
          aria-selected={i === selectedIndex}
          onClick={() => select(item)}
          onMouseEnter={() => setSelectedIndex(i)}
          className={cn(
            'flex items-center gap-3 w-full px-3 py-2 text-left transition-colors',
            i === selectedIndex ? 'bg-accent/80 text-accent-foreground' : 'text-popover-foreground hover:bg-accent/40',
          )}
        >
          <span className={cn(
            'text-[10px] font-medium px-1.5 py-0.5 rounded shrink-0',
            item.type === 'agent' ? 'bg-primary/15 text-primary' : 'bg-success/15 text-success'
          )}>
            {item.type === 'agent' ? 'Agent' : 'Model'}
          </span>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{item.name}</p>
            {item.description && (
              <p className="text-[10px] text-muted-foreground/60 truncate mt-0.5">{item.description}</p>
            )}
          </div>
          <span className="text-[10px] text-muted-foreground/40 shrink-0 tabular-nums">
            {i + 1}/{scored.length}
          </span>
        </button>
      ))}
    </div>
  )
}
