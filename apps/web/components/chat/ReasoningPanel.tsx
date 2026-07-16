'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'

interface ReasoningPanelProps {
  isThinking: boolean
  className?: string
}

export function ReasoningPanel({ isThinking, className }: ReasoningPanelProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className={cn("mx-auto w-full max-w-2xl px-3 sm:px-4", className)}>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-2 rounded-lg border transition-colors text-left",
          isThinking
            ? "border-primary/20 bg-primary/[0.03]"
            : "border-border/30 bg-muted/20 hover:bg-muted/30 text-muted-foreground",
        )}
        aria-expanded={expanded}
        aria-label={expanded ? 'Hide reasoning' : 'Show reasoning'}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <svg className={cn("w-4 h-4 shrink-0", isThinking ? "text-primary" : "text-muted-foreground")} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
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
        <svg
          className={cn("w-3.5 h-3.5 shrink-0 transition-transform", expanded && "rotate-180")}
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && isThinking && (
        <div className="mt-1 px-3 py-2 rounded-lg border border-border/20 bg-muted/10 text-[11px] text-muted-foreground leading-relaxed">
          Generating response with contextual understanding of conversation history, model parameters, and applied knowledge signals.
        </div>
      )}
    </div>
  )
}
