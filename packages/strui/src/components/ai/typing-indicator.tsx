import * as React from 'react'

import { cn } from '../../lib/cn'

export interface TypingIndicatorProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string
}

/** Streaming / “assistant is typing” affordance — works without motion for a11y when prefers-reduced-motion. */
export function TypingIndicator({ className, label = 'Assistant is responding', ...props }: TypingIndicatorProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={label}
      className={cn(
        'mr-auto flex w-fit items-center gap-2 px-0 py-3 text-muted-foreground',
        className,
      )}
      {...props}
    >
      <span className="sr-only">{label}</span>
      <span className="flex gap-1" aria-hidden>
        <span className="str-dot str-dot-1" />
        <span className="str-dot str-dot-2" />
        <span className="str-dot str-dot-3" />
      </span>
    </div>
  )
}
