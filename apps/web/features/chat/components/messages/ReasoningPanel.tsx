'use client'

import { useState, memo, useRef, useEffect } from 'react'
import { cn, IconBrain, IconChevronDown } from '@sloughgpt/strui'

interface ContextLayer {
  type: 'knowledge' | 'memory' | 'rag' | 'tool' | 'soul' | 'system'
  label: string
  detail?: string
}

const LAYER_STYLES: Record<ContextLayer['type'], { dot: string; badge: string }> = {
  knowledge: { dot: 'bg-primary', badge: 'bg-primary/10 text-primary/80 border-primary/15' },
  memory:    { dot: 'bg-accent', badge: 'bg-accent/10 text-accent-foreground/80 border-accent/15' },
  rag:       { dot: 'bg-success', badge: 'bg-success/10 text-success border-success/15' },
  tool:      { dot: 'bg-warning', badge: 'bg-warning/10 text-warning border-warning/15' },
  soul:      { dot: 'bg-primary/60', badge: 'bg-primary/5 text-primary/70 border-primary/10' },
  system:    { dot: 'bg-muted-foreground/60', badge: 'bg-muted text-muted-foreground border-border/40' },
}

interface ReasoningPanelProps {
  isThinking: boolean
  contextLayers?: ContextLayer[]
  className?: string
}

export const ReasoningPanel = memo(function ReasoningPanel({ isThinking, contextLayers = [], className }: ReasoningPanelProps) {
  const [expanded, setExpanded] = useState(false)
  const contentRef = useRef<HTMLDivElement>(null)
  const [contentHeight, setContentHeight] = useState(0)

  useEffect(() => {
    if (contentRef.current) {
      setContentHeight(contentRef.current.scrollHeight)
    }
  }, [expanded, contextLayers, isThinking])

  return (
    <div className={cn("mx-auto w-full max-w-4xl px-4 sm:px-6", className)}>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-all duration-200 text-left group",
          isThinking
            ? "border-primary/20 bg-primary/[0.04] hover:bg-primary/[0.06]"
            : "border-border/30 bg-muted/15 hover:bg-muted/25 hover:border-border/40 text-muted-foreground",
        )}
        aria-expanded={expanded}
        aria-label={expanded ? 'Hide reasoning' : 'Show reasoning'}
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className={cn(
            "relative flex h-4 w-4 shrink-0 items-center justify-center",
            isThinking && "animate-[pulse_2s_ease-in-out_infinite]"
          )} aria-hidden="true">
            <IconBrain className={cn("h-4 w-4", isThinking ? "text-primary" : "text-muted-foreground/60")} />
            {isThinking && (
              <span className="absolute inset-0 rounded-full bg-primary/20 animate-ping [animation-duration:1.5s]" />
            )}
          </span>
          <span className={cn("text-xs font-medium", isThinking ? "text-foreground" : "text-muted-foreground")}>
            {isThinking ? 'Reasoning' : 'Reasoning complete'}
          </span>
          {isThinking && (
            <span className="flex gap-0.5 ml-0.5" aria-hidden="true">
              <span className="w-1 h-1 rounded-full bg-primary/70 animate-bounce [animation-delay:0ms]" />
              <span className="w-1 h-1 rounded-full bg-primary/70 animate-bounce [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-primary/70 animate-bounce [animation-delay:300ms]" />
            </span>
          )}
          {contextLayers.length > 0 && !isThinking && (
            <span className="inline-flex items-center gap-1 ml-1">
              {contextLayers.map((layer, i) => {
                const styles = LAYER_STYLES[layer.type]
                return (
                  <span
                    key={i}
                    className={cn("inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full border text-[9px] font-medium", styles.badge)}
                    title={layer.detail}
                  >
                    <span className={cn("w-1 h-1 rounded-full", styles.dot)} />
                    {layer.label}
                  </span>
                )
              })}
            </span>
          )}
        </div>
        <IconChevronDown
          className={cn(
            "w-3.5 h-3.5 shrink-0 transition-transform duration-200",
            expanded && "rotate-180",
            isThinking ? "text-primary/60" : "text-muted-foreground/40 group-hover:text-muted-foreground/60"
          )}
          aria-hidden="true"
        />
      </button>

      <div
        className="overflow-hidden transition-[grid-template-rows] duration-300 ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ gridTemplateRows: expanded ? '1fr' : '0fr' }}
        role="region"
        aria-hidden={!expanded}
      >
        <div ref={contentRef} className="min-h-0">
          {expanded && (
            <div className="mt-1.5 px-3 py-2.5 rounded-lg border border-border/25 bg-muted/10">
              {contextLayers.length > 0 ? (
                <div className="space-y-1">
                  {contextLayers.map((layer, i) => {
                    const styles = LAYER_STYLES[layer.type]
                    return (
                      <div key={i} className="flex items-start gap-2 text-[11px] leading-relaxed">
                        <span className={cn("shrink-0 mt-1 w-1.5 h-1.5 rounded-full", styles.dot)} />
                        <div className="flex-1 min-w-0">
                          <span className="font-medium text-foreground/80">{layer.label}</span>
                          {layer.detail && (
                            <span className="text-muted-foreground/50 ml-1.5 font-mono text-[10px]">{layer.detail}</span>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <p className="text-[11px] text-muted-foreground/60 leading-relaxed">
                  {isThinking
                    ? 'Generating response with contextual understanding of conversation history, model parameters, and applied knowledge signals.'
                    : 'No context layers were applied for this response.'}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
})
