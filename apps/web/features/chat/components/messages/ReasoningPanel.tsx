'use client'

import { useState } from 'react'
import { cn, IconBrain, IconChevronDown } from '@sloughgpt/strui'

interface ContextLayer {
  type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'
  label: string
  detail?: string
}

interface ReasoningPanelProps {
  isThinking: boolean
  contextLayers?: ContextLayer[]
  className?: string
}

export function ReasoningPanel({ isThinking, contextLayers = [], className }: ReasoningPanelProps) {
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
          {contextLayers.length > 0 && !isThinking && (
            <span className="text-[10px] text-muted-foreground/60 ml-1">
              {contextLayers.length} context {contextLayers.length === 1 ? 'layer' : 'layers'}
            </span>
          )}
        </div>
        <IconChevronDown className={cn("w-3.5 h-3.5 shrink-0 transition-transform", expanded && "rotate-180")} aria-hidden="true" />
      </button>

      {expanded && (
        <div className="mt-1 px-3 py-2 rounded-lg border border-border/20 bg-muted/10 space-y-1.5">
          {contextLayers.length > 0 ? (
            contextLayers.map((layer, i) => (
              <div key={i} className="flex items-start gap-2 text-[11px] leading-relaxed">
                <span className={cn(
                  "shrink-0 mt-0.5 w-1.5 h-1.5 rounded-full",
                  layer.type === 'knowledge' && "bg-blue-500",
                  layer.type === 'memory' && "bg-purple-500",
                  layer.type === 'rag' && "bg-green-500",
                  layer.type === 'tool' && "bg-orange-500",
                  layer.type === 'soul' && "bg-primary",
                  layer.type === 'system' && "bg-muted-foreground",
                )} />
                <div className="flex-1 min-w-0">
                  <span className="font-medium text-foreground/80">{layer.label}</span>
                  {layer.detail && (
                    <span className="text-muted-foreground/70 ml-1.5">{layer.detail}</span>
                  )}
                </div>
              </div>
            ))
          ) : (
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              {isThinking
                ? 'Generating response with contextual understanding of conversation history, model parameters, and applied knowledge signals.'
                : 'No context layers were applied for this response.'}
            </p>
          )}
        </div>
      )}
    </div>
  )
}
