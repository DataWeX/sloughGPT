'use client'

import { useState, useCallback, memo } from 'react'
import { cn } from '@sloughgpt/strui'

export interface ContextInjectionBarProps {
  onInject: (context: string) => void
  disabled?: boolean
  className?: string
}

export const ContextInjectionBar = memo(function ContextInjectionBar({
  onInject,
  disabled = false,
  className,
}: ContextInjectionBarProps) {
  const [context, setContext] = useState('')
  const [showInput, setShowInput] = useState(false)

  const handleInject = useCallback(() => {
    if (context.trim()) {
      onInject(context.trim())
      setContext('')
      setShowInput(false)
    }
  }, [context, onInject])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleInject()
    } else if (e.key === 'Escape') {
      setShowInput(false)
      setContext('')
    }
  }, [handleInject])

  if (!showInput) {
    return (
      <button
        type="button"
        onClick={() => setShowInput(true)}
        disabled={disabled}
        className={cn(
          'text-xs text-muted-foreground hover:text-foreground transition-colors',
          disabled && 'opacity-50 cursor-not-allowed',
          className
        )}
        aria-label="Inject context"
      >
        + Context
      </button>
    )
  }

  return (
    <div className={cn('flex items-center gap-2', className)}>
      <input
        type="text"
        value={context}
        onChange={(e) => setContext(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Add context..."
        className="flex-1 text-xs bg-muted/50 border border-border rounded-lg px-2 py-1 focus:outline-none focus:ring-1 focus:ring-primary"
        autoFocus
      />
      <button
        type="button"
        onClick={handleInject}
        disabled={!context.trim()}
        className="text-xs text-primary hover:text-primary/80 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Send
      </button>
      <button
        type="button"
        onClick={() => { setShowInput(false); setContext('') }}
        className="text-xs text-muted-foreground hover:text-foreground"
      >
        Cancel
      </button>
    </div>
  )
})
