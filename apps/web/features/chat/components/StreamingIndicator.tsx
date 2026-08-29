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
    label: 'Thinking...',
    color: 'text-primary',
    dots: true,
  },
  generating: {
    label: 'Generating...',
    color: 'text-primary',
    dots: true,
  },
  tool_call: {
    label: 'Running tool...',
    color: 'text-warning',
    dots: false,
  },
  context: {
    label: 'Processing context...',
    color: 'text-muted-foreground',
    dots: true,
  },
  error: {
    label: 'Error occurred',
    color: 'text-destructive',
    dots: false,
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
        'flex items-center gap-2 text-xs',
        config.color,
        className
      )}
      role="status"
      aria-live="polite"
    >
      {config.dots && (
        <div className="flex gap-1">
          <span className="w-1 h-1 rounded-full bg-current animate-bounce [animation-delay:0ms]" />
          <span className="w-1 h-1 rounded-full bg-current animate-bounce [animation-delay:150ms]" />
          <span className="w-1 h-1 rounded-full bg-current animate-bounce [animation-delay:300ms]" />
        </div>
      )}
      <span>
        {config.label}
        {toolName && status === 'tool_call' && (
          <span className="text-muted-foreground ml-1">({toolName})</span>
        )}
      </span>
    </div>
  )
})
