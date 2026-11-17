'use client'

import { cn } from '@sloughgpt/strui'

interface StreamingCursorProps {
  className?: string
}

/**
 * StreamingCursor — the single "generation in progress" caret.
 *
 * One state-derived visual, one definition: the Quick test probe and the chat
 * message stream both render this caret while their stream is live, so changing
 * its look or its motion behavior is one edit instead of three drifting copies
 * (card 20261117_123807).
 *
 * `aria-hidden` is deliberate — the message frame already carries
 * `aria-live="polite"` while streaming; a second announced node would double
 * every token. `motion-reduce:animate-none` keeps it still for users who asked
 * for reduced motion.
 */
export function StreamingCursor({ className }: StreamingCursorProps) {
  return (
    <span
      className={cn(
        'inline-block w-0.5 h-4 bg-primary ml-0.5 align-middle animate-pulse motion-reduce:animate-none',
        className,
      )}
      aria-hidden="true"
    />
  )
}
