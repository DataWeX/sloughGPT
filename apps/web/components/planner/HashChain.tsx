'use client'

import { useState, useEffect } from 'react'
import { cn } from '@sloughgpt/strui'
import { fetchHashTree } from '@/lib/planner-client'
import type { HashTree } from './types'

interface HashChainProps {
  cardId: string
  /** Bump to refetch after a note mutation */
  refreshKey?: number
}

const CHANGE_COLOR: Record<string, string> = {
  note_add: 'text-success',
  note_edit: 'text-warning',
  note_delete: 'text-destructive',
  rehash: 'text-info',
}

export function HashChain({ cardId, refreshKey = 0 }: HashChainProps) {
  const [tree, setTree] = useState<HashTree | null>(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    let alive = true
    fetchHashTree(cardId)
      .then((t) => {
        if (alive) setTree(t)
      })
      .catch(() => {
        if (alive) setTree(null)
      })
    return () => {
      alive = false
    }
  }, [cardId, refreshKey])

  if (!tree) return null

  const short = (h: string) => (h ? h.slice(0, 10) : '—')

  return (
    <div className="rounded-lg border border-border bg-card/50 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
          Hash Chain
        </span>
        <span className="text-[10px] text-muted-foreground font-mono">
          {tree.history.length} {tree.history.length === 1 ? 'change' : 'changes'}
        </span>
      </div>

      <div className="flex items-center gap-3 text-[11px] font-mono">
        <span className="text-muted-foreground">root</span>
        <span className="text-muted-foreground">{short(tree.root.root)}</span>
        <span className="text-muted-foreground">slot</span>
        <span className="text-muted-foreground">{short(tree.root.slot_id)}</span>
      </div>

      {tree.notes.length > 0 && (
        <div className="space-y-0.5">
          {tree.notes.map((n) => (
            <div key={n.note_id} className="flex items-center gap-3 text-[11px] font-mono">
              <span className="text-muted-foreground">note</span>
              <span className="text-muted-foreground">{short(n.hash_value)}</span>
              <span className="text-muted-foreground">v{n.version}</span>
            </div>
          ))}
        </div>
      )}

      {tree.history.length > 0 && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[10px] font-mono text-muted-foreground hover:text-muted-foreground transition-colors"
          >
            {expanded ? 'hide chain' : 'show chain'}
          </button>

          {expanded && (
            <div className="space-y-0.5 border-t border-border pt-2">
              {tree.history.map((h, i) => (
                <div key={i} className="flex items-center gap-1.5 text-[10px] font-mono">
                  <span className="text-muted-foreground w-4">{i}</span>
                  <span className="text-muted-foreground">{short(h.old_hash)}</span>
                  <span className="text-muted-foreground">→</span>
                  <span className="text-muted-foreground">{short(h.new_hash)}</span>
                  <span className={cn('ml-auto', CHANGE_COLOR[h.change_type] || 'text-muted-foreground')}>
                    {h.change_type}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
