'use client'

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { cn } from '@/lib/cn'
import { getAllCommands } from '@/lib/chat-commands'
import type { ChatCommand } from '@/lib/chat-commands'

interface SlashCommandMenuProps {
  value: string
  onInsert: (text: string) => void
  onClose: () => void
  onExecute?: (cmd: ChatCommand, args: string[]) => void
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

function parseArgs(value: string, commandName: string): string[] {
  const trimmed = value.trim()
  const parts = trimmed.split(/\s+/)
  const rest = parts.slice(1).filter(Boolean)
  return rest
}

export function SlashCommandMenu({ value, onInsert, onClose, onExecute }: SlashCommandMenuProps) {
  const allCommands = useMemo(() => getAllCommands(), [])

  const query = value.startsWith('/') ? value.slice(1) : ''
  const firstWord = query.split(/\s+/)[0] || ''

  const scored = useMemo(() => {
    if (!query) return allCommands.map(c => ({ command: c, score: 0 }))
    return allCommands
      .map(c => {
        const nameScore = fuzzyScore(firstWord, c.command.slice(1))
        const descScore = fuzzyScore(query, c.description)
        const score = Math.max(nameScore, descScore ?? -1)
        return { command: c, score }
      })
      .filter(x => x.score >= 0)
      .sort((a, b) => b.score - a.score)
  }, [query, firstWord, allCommands])

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

  const select = useCallback((cmd: ChatCommand) => {
    if (onExecute) {
      const args = parseArgs(value, cmd.command)
      onExecute(cmd, args)
    } else {
      onInsert(cmd.command)
    }
    onClose()
  }, [onInsert, onClose, onExecute, value])

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
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
        if (scored[selectedIndex]) select(scored[selectedIndex].command)
        break
      case 'Escape':
        e.preventDefault()
        onClose()
        break
    }
  }, [scored, selectedIndex, select, onClose])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  if (scored.length === 0) return null

  return (
    <div
      ref={listRef}
      role="listbox"
      aria-label="Slash commands"
      className="absolute bottom-full left-0 right-0 mb-1 mx-1 max-h-64 overflow-y-auto rounded-lg border border-border/40 bg-popover/95 backdrop-blur-sm shadow-xl z-50"
    >
      {scored.map(({ command: cmd, score }, i) => (
        <button
          key={cmd.command}
          ref={el => { itemRefs.current[i] = el }}
          role="option"
          aria-selected={i === selectedIndex}
          onClick={() => select(cmd)}
          onMouseEnter={() => setSelectedIndex(i)}
          className={cn(
            'flex items-center gap-3 w-full px-3 py-2 text-left transition-colors',
            i === selectedIndex ? 'bg-accent/80 text-accent-foreground' : 'text-popover-foreground hover:bg-accent/40',
          )}
        >
          <code className="text-sm font-medium shrink-0 min-w-[5rem] text-primary/80">
            {cmd.command}
          </code>
          <div className="flex-1 min-w-0">
            <p className="text-xs truncate">{cmd.description}</p>
            {cmd.usage !== cmd.command && (
              <p className="text-[10px] text-muted-foreground/60 truncate mt-0.5">{cmd.usage}</p>
            )}
          </div>
          <span className="text-[10px] text-muted-foreground/40 shrink-0 tabular-nums">
            {Math.round(score)}%
          </span>
        </button>
      ))}
    </div>
  )
}
