'use client'

import { useState, useMemo, useCallback, useEffect, useRef } from 'react'
import { findMatchingCommands, type SlashCommand } from '@/lib/slash-commands'

interface SlashCommandMenuProps {
  query: string
  visible: boolean
  onSelect: (command: SlashCommand) => void
  onClose: () => void
}

export function SlashCommandMenu({ query, visible, onSelect, onClose }: SlashCommandMenuProps) {
  const [selectedIndex, setSelectedIndex] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const matches = useMemo(() => findMatchingCommands(query), [query])

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!visible) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setSelectedIndex(i => Math.min(i + 1, matches.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setSelectedIndex(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter' || e.key === 'Tab') {
      e.preventDefault()
      if (matches[selectedIndex]) {
        onSelect(matches[selectedIndex])
      }
    } else if (e.key === 'Escape') {
      e.preventDefault()
      onClose()
    }
  }, [visible, matches, selectedIndex, onSelect, onClose])

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  if (!visible || matches.length === 0) return null

  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} />
      <div
        ref={containerRef}
        className="absolute bottom-full left-0 right-0 z-50 mb-1 max-h-48 overflow-y-auto rounded-lg border border-border/40 bg-popover p-1 shadow-lg"
        role="listbox"
      >
        {matches.map((cmd, i) => (
          <button
            key={cmd.name}
            role="option"
            aria-selected={i === selectedIndex}
            className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm transition-colors ${
              i === selectedIndex ? 'bg-primary/10 text-primary' : 'hover:bg-muted'
            }`}
            onClick={() => onSelect(cmd)}
            onMouseEnter={() => setSelectedIndex(i)}
          >
            <span className="text-xs">{cmd.icon}</span>
            <span className="font-medium">/{cmd.name}</span>
            <span className="text-xs text-muted-foreground">{cmd.description}</span>
          </button>
        ))}
      </div>
    </>
  )
}
