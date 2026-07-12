'use client'

import { forwardRef, useRef, type TextareaHTMLAttributes } from 'react'
import { cn } from '../../lib/cn'

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Auto-grow to fit content up to maxRows */
  autoResize?: boolean
  /** Max rows before scrolling (only with autoResize) */
  maxRows?: number
  /** Error styling */
  error?: boolean
}

const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, autoResize = false, maxRows = 8, error, onInput, ...props }, ref) => {
    const lineHeight = 20 // px, matches text-sm + py-2
    const minHeight = 80 // px, min-h-20

    const handleInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
      if (autoResize) {
        const el = e.currentTarget
        el.style.height = 'auto'
        const capped = Math.min(el.scrollHeight, maxRows * lineHeight + 16)
        el.style.height = `${capped}px`
      }
      onInput?.(e)
    }

    return (
      <textarea
        ref={ref}
        onInput={handleInput}
        className={cn(
          'flex w-full rounded-lg border bg-background px-3 py-2 text-sm text-foreground shadow-sm',
          'transition-[border-color,box-shadow,background-color] duration-200',
          'placeholder:text-muted-foreground selection:bg-primary/20 selection:text-foreground',
          'hover:border-primary/50',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background focus-visible:border-primary/60',
          'disabled:cursor-not-allowed disabled:opacity-50',
          autoResize ? 'resize-none overflow-hidden' : 'resize-y',
          error
            ? 'border-destructive/60 focus-visible:ring-destructive/40 hover:border-destructive/70'
            : 'border-input',
          className,
        )}
        style={autoResize ? { minHeight } : { minHeight }}
        {...props}
      />
    )
  },
)
Textarea.displayName = 'Textarea'

export { Textarea }
