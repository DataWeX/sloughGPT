'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconX, IconPlus } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

interface TagInputProps {
  tags: string[]
  onAdd: (tag: string) => void
  onRemove: (tag: string) => void
  placeholder?: string
  className?: string
  disabled?: boolean
}

export const TagInput = memo(function TagInput({
  tags,
  onAdd,
  onRemove,
  placeholder = 'Add tag...',
  className,
  disabled = false,
}: TagInputProps) {
  const [input, setInput] = useState('')

  const handleAdd = useCallback(() => {
    const trimmed = input.trim()
    if (trimmed && !tags.includes(trimmed.toLowerCase())) {
      onAdd(trimmed)
      setInput('')
    }
  }, [input, tags, onAdd])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleAdd()
    }
    if (e.key === 'Backspace' && !input && tags.length > 0) {
      onRemove(tags[tags.length - 1])
    }
  }, [handleAdd, input, tags, onRemove])

  return (
    <div className={cn('flex flex-wrap items-center gap-1', className)}>
      {tags.map(tag => (
        <span
          key={tag}
          className="inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-medium bg-primary/10 text-primary rounded border border-primary/20"
        >
          {tag}
          {!disabled && (
            <button
              type="button"
              onClick={() => onRemove(tag)}
              className="hover:text-primary/70"
              aria-label={`Remove ${tag}`}
            >
              <IconX className="h-2.5 w-2.5" />
            </button>
          )}
        </span>
      ))}
      {!disabled && (
        <div className="flex items-center gap-1">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={tags.length === 0 ? placeholder : ''}
            className="w-16 text-[10px] bg-transparent border-0 p-0 focus:outline-none focus:ring-0 placeholder:text-muted-foreground/50"
          />
          {input.trim() && (
            <Button
              variant="ghost"
              size="icon-sm"
              className="h-4 w-4"
              onClick={handleAdd}
              aria-label="Add tag"
            >
              <IconPlus className="h-2.5 w-2.5" />
            </Button>
          )}
        </div>
      )}
    </div>
  )
})