'use client'

import { useState } from 'react'
import { cn, IconBrain, IconChevronDown } from '@sloughgpt/strui'

interface ReasoningPanelProps {
  isThinking: boolean
  className?: string
}

export function ReasoningPanel({ isThinking, className }: ReasoningPanelProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={cn("mx-auto w-full max-w-3xl px-4 sm:px-6", className)}>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors text-left",
          isThinking
            ? "border-primary/20 bg-primary/[0.03]"
            : "border-border/30 bg-muted/20 hover:bg-muted/30 text-muted-foreground",
        )}
        aria-expanded={expanded}
        aria-label={expanded ? 'Hide reasoning' : 'Show reasoning'}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <IconBrain className={cn("w-4 h-4 shrink-0", isThinking ? "text-primary" : "text-muted-foreground")} aria-hidden="true" />
          <span className="text-xs font-medium">
            {isThinking ? 'Reasoning' : 'Reasoning complete'}
          </span>
          {isThinking && (
            <span className="flex gap-0.5" aria-hidden="true">
              <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:0ms]" />
              <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-primary animate-bounce [animation-delay:300ms]" />
            </span>
          )}
        </div>
        <IconChevronDown className={cn("w-3.5 h-3.5 shrink-0 transition-transform", expanded && "rotate-180")} aria-hidden="true" />
      </button>

      {expanded && isThinking && (
        <div className="mt-1 px-3 py-2 rounded-lg border border-border/20 bg-muted/10 text-[11px] text-muted-foreground leading-relaxed">
          Generating response with contextual understanding of conversation history, model parameters, and applied knowledge signals.
        </div>
      )}
    </div>
  )
}
