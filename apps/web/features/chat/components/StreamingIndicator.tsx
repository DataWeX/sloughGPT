'use client'

import { memo } from 'react'
import { cn } from '@sloughgpt/strui'

export interface StreamingIndicatorProps {
  status: 'thinking' | 'generating' | 'tool_call' | 'context' | 'error'
  toolName?: string
  className?: string
}

const STATUS_CONFIG = {
  thinking: {
    label: 'Thinking',
    color: 'text-primary',
  },
  generating: {
    label: 'Generating',
    color: 'text-primary',
  },
  tool_call: {
    label: 'Running tool',
    color: 'text-warning',
  },
  context: {
    label: 'Processing',
    color: 'text-muted-foreground',
  },
  error: {
    label: 'Error',
    color: 'text-destructive',
  },
} as const

export const StreamingIndicator = memo(function StreamingIndicator({
  status,
  toolName,
  className,
}: StreamingIndicatorProps) {
  const config = STATUS_CONFIG[status]

  return (
    <div
      className={cn(
        'flex items-center gap-1.5 text-xs',
        config.color,
        className
      )}
      role="status"
      aria-live="polite"
    >
      <span className="flex gap-0.5">
        <span className="w-1 h-1 rounded-full bg-current animate-bounce [animation-delay:0ms]" />
        <span className="w-1 h-1 rounded-full bg-current animate-bounce [animation-delay:150ms]" />
        <span className="w-1 h-1 rounded-full bg-current animate-bounce [animation-delay:300ms]" />
      </span>
      <span className="text-muted-foreground">
        {config.label}
        {toolName && status === 'tool_call' && (
          <span className="ml-0.5 opacity-60">({toolName})</span>
        )}
      </span>
    </div>
  )
})
