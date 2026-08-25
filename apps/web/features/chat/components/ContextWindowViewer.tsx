'use client'

import { useState, useCallback, memo } from 'react'
import { Button, IconChevronDown, IconChevronRight, IconCopy, IconCheck } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'

export interface ContextWindowItem {
  label: string
  content: string
  tokenCount?: number
  type: 'system' | 'knowledge' | 'agent' | 'user' | 'assistant'
}

interface ContextWindowViewerProps {
  items: ContextWindowItem[]
  totalTokens?: number
  className?: string
  onRefresh?: () => void
}

const TYPE_COLORS: Record<string, string> = {
  system: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
  knowledge: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  agent: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  user: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  assistant: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
}

const TYPE_LABELS: Record<string, string> = {
  system: 'System Prompt',
  knowledge: 'Knowledge Base',
  agent: 'Agent Instructions',
  user: 'User Message',
  assistant: 'Assistant Response',
}

function ContextWindowItemCard({ item, isExpanded, onToggle }: {
  item: ContextWindowItem
  isExpanded: boolean
  onToggle: () => void
}) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(item.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }, [item.content])

  return (
    <div className="border border-border/30 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-muted/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className={cn(
            'text-[10px] font-medium px-1.5 py-0.5 rounded border',
            TYPE_COLORS[item.type]
          )}>
            {TYPE_LABELS[item.type]}
          </span>
          <span className="text-xs text-foreground/80">{item.label}</span>
        </div>
        <div className="flex items-center gap-2">
          {item.tokenCount && (
            <span className="text-[10px] text-muted-foreground">
              ~{item.tokenCount} tokens
            </span>
          )}
          {isExpanded ? (
            <IconChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <IconChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
        </div>
      </button>
      
      {isExpanded && (
        <div className="px-3 pb-3">
          <div className="relative">
            <pre className="text-xs text-foreground/70 bg-muted/30 rounded p-2 overflow-x-auto max-h-48 overflow-y-auto whitespace-pre-wrap">
              {item.content}
            </pre>
            <Button
              variant="ghost"
              size="icon-sm"
              className="absolute top-1 right-1 h-6 w-6"
              onClick={handleCopy}
              aria-label={copied ? 'Copied' : 'Copy content'}
            >
              {copied ? <IconCheck className="h-3 w-3" /> : <IconCopy className="h-3 w-3" />}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export const ContextWindowViewer = memo(function ContextWindowViewer({
  items,
  totalTokens,
  className,
  onRefresh,
}: ContextWindowViewerProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set())
  const [showAll, setShowAll] = useState(false)

  const toggleItem = useCallback((label: string) => {
    setExpandedItems(prev => {
      const next = new Set(prev)
      if (next.has(label)) {
        next.delete(label)
      } else {
        next.add(label)
      }
      return next
    })
  }, [])

  const toggleAll = useCallback(() => {
    if (showAll) {
      setExpandedItems(new Set())
    } else {
      setExpandedItems(new Set(items.map(i => i.label)))
    }
    setShowAll(!showAll)
  }, [showAll, items])

  if (items.length === 0) {
    return (
      <div className={cn('text-center py-4 text-muted-foreground text-xs', className)}>
        No context items loaded
      </div>
    )
  }

  const estimatedTokens = totalTokens ?? items.reduce((sum, item) => sum + (item.tokenCount ?? Math.ceil(item.content.length / 4)), 0)

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-foreground/80">
            Context Window
          </span>
          <span className="text-[10px] text-muted-foreground">
            {items.length} items · ~{estimatedTokens.toLocaleString()} tokens
          </span>
        </div>
        <div className="flex items-center gap-1">
          {onRefresh && (
            <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={onRefresh}>
              Refresh
            </Button>
          )}
          <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={toggleAll}>
            {showAll ? 'Collapse All' : 'Expand All'}
          </Button>
        </div>
      </div>
      
      <div className="space-y-1">
        {items.map(item => (
          <ContextWindowItemCard
            key={item.label}
            item={item}
            isExpanded={expandedItems.has(item.label)}
            onToggle={() => toggleItem(item.label)}
          />
        ))}
      </div>
    </div>
  )
})