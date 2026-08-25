'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { cn } from '@sloughgpt/strui'

interface MessageReactionsProps {
  reactions?: Record<string, string[]>
  onReact: (emoji: string) => void
  className?: string
}

const QUICK_REACTIONS = ['👍', '❤️', '😊', '🤔', '👏', '🔥']

export function MessageReactions({ reactions = {}, onReact, className }: MessageReactionsProps) {
  const [showPicker, setShowPicker] = useState(false)
  const pickerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!showPicker) return
    const handleClickOutside = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setShowPicker(false)
      }
    }
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowPicker(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    document.addEventListener('keydown', handleEscape)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
      document.removeEventListener('keydown', handleEscape)
    }
  }, [showPicker])

  const handleReact = useCallback((emoji: string) => {
    onReact(emoji)
    setShowPicker(false)
  }, [onReact])

  const entries = Object.entries(reactions).filter(([, users]) => users.length > 0)

  return (
    <div className={cn("flex items-center gap-1 flex-wrap", className)}>
      {entries.map(([emoji, users]) => (
        <button
          key={emoji}
          type="button"
          onClick={() => handleReact(emoji)}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[11px] bg-muted/50 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
          aria-label={`${emoji} reaction, ${users.length} ${users.length === 1 ? 'person' : 'people'}`}
        >
          <span>{emoji}</span>
          {users.length > 1 && <span className="tabular-nums">{users.length}</span>}
        </button>
      ))}
      
      <div className="relative" ref={pickerRef}>
        <button
          type="button"
          onClick={() => setShowPicker(!showPicker)}
          className="inline-flex items-center justify-center w-6 h-6 rounded-full text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/50 transition-colors opacity-0 group-hover:opacity-100"
          aria-label="Add reaction"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M8 14s1.5 2 4 2 4-2 4-2" />
            <line x1="9" y1="9" x2="9.01" y2="9" />
            <line x1="15" y1="9" x2="15.01" y2="9" />
          </svg>
        </button>

        {showPicker && (
          <div className="absolute bottom-full left-0 mb-1 p-1.5 rounded-lg bg-card border border-border/50 shadow-lg flex gap-1 z-50">
            {QUICK_REACTIONS.map(emoji => (
              <button
                key={emoji}
                type="button"
                onClick={() => handleReact(emoji)}
                className="w-8 h-8 flex items-center justify-center rounded hover:bg-muted text-lg transition-colors"
                aria-label={`React with ${emoji}`}
              >
                {emoji}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}