'use client'

import { useState, useCallback, useMemo, useEffect, useRef, memo, type ReactNode } from 'react'
import { Button } from '@sloughgpt/strui'
import { cn, IconExplain, IconDocument, IconCog, IconEye } from '@sloughgpt/strui'
import { estimateTokens } from '@/lib/format-bytes'

interface AgentStats {
  tokens: number
  words: number
  chars: number
  lines: number
  estimatedCost: number
}

interface QuickAction {
  id: string
  label: string
  icon: ReactNode
  action: string
  category: 'prompt' | 'format' | 'context' | 'tool'
}

interface ChatInputAgentBoxProps {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  onInsertAction: (action: string) => void
  model?: string
  tier?: 'free' | 'pro' | 'enterprise'
  contextTokens?: number
  maxContext?: number
  className?: string
}

const MODEL_PRICING: Record<string, { input: number; output: number }> = {
  'gpt-4': { input: 0.03, output: 0.06 },
  'gpt-4-turbo': { input: 0.01, output: 0.03 },
  'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
  'claude-3-opus': { input: 0.015, output: 0.075 },
  'claude-3-sonnet': { input: 0.003, output: 0.015 },
}

const QUICK_ACTIONS: QuickAction[] = [
  { id: 'explain', label: 'Explain', icon: <IconExplain className="h-4 w-4" />, action: 'Explain this concept in simple terms:', category: 'prompt' },
  { id: 'summarize', label: 'Summarize', icon: <IconDocument className="h-4 w-4" />, action: 'Summarize the key points:', category: 'prompt' },
  { id: 'debug', label: 'Debug', icon: <IconCog className="h-4 w-4" />, action: 'Help me debug this code:', category: 'prompt' },
  { id: 'review', label: 'Review', icon: <IconEye className="h-4 w-4" />, action: 'Review this code and suggest improvements:', category: 'prompt' },
  { id: 'bold', label: 'Bold', icon: '**', action: '**text**', category: 'format' },
  { id: 'italic', label: 'Italic', icon: '_', action: '_text_', category: 'format' },
  { id: 'code', label: 'Code', icon: '`', action: '`code`', category: 'format' },
  { id: 'list', label: 'List', icon: '•', action: '\n- ', category: 'format' },
  { id: 'context', label: 'Add Context', icon: '+', action: '[Context: ]', category: 'context' },
  { id: 'reference', label: 'Reference', icon: '@', action: '@reference ', category: 'context' },
]

export const ChatInputAgentBox = memo(function ChatInputAgentBox({
  value,
  onChange,
  onSend,
  onInsertAction,
  model = 'gpt-4',
  tier = 'free',
  contextTokens = 0,
  maxContext = 128000,
  className,
}: ChatInputAgentBoxProps) {
  const [showActions, setShowActions] = useState(false)
  const [actionCategory, setActionCategory] = useState<string | null>(null)
  const [showStats, setShowStats] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const stats = useMemo((): AgentStats => {
    const tokens = estimateTokens(value)
    const words = value.trim() ? value.trim().split(/\s+/).length : 0
    const chars = value.length
    const lines = value.split('\n').length
    const pricing = MODEL_PRICING[model] || MODEL_PRICING['gpt-4']
    const estimatedCost = (tokens / 1000) * pricing.input

    return { tokens, words, chars, lines, estimatedCost }
  }, [value, model])

  const contextUsage = useMemo(() => {
    const percentage = maxContext > 0 ? (contextTokens / maxContext) * 100 : 0
    return {
      percentage: Math.min(100, percentage),
      remaining: Math.max(0, maxContext - contextTokens),
      isWarning: percentage > 80,
      isCritical: percentage > 95,
    }
  }, [contextTokens, maxContext])

  const filteredActions = useMemo(() => {
    return actionCategory
      ? QUICK_ACTIONS.filter(a => a.category === actionCategory)
      : QUICK_ACTIONS
  }, [actionCategory])

  const handleInsert = useCallback((action: string) => {
    onInsertAction(action)
    setShowActions(false)
    textareaRef.current?.focus()
  }, [onInsertAction])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      onSend()
    }
    if (e.key === 'Escape') {
      setShowActions(false)
    }
  }, [onSend])

  const formatCost = (cost: number) => {
    if (cost < 0.001) return '<$0.001'
    return `$${cost.toFixed(4)}`
  }

  return (
    <div className={cn('border rounded-xl bg-card overflow-hidden', className)}>
      {/* Header with context indicator */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <div className={cn(
            'w-2 h-2 rounded-full',
            contextUsage.isCritical ? 'bg-destructive animate-pulse' :
            contextUsage.isWarning ? 'bg-warning' : 'bg-success',
          )} />
          <span className="text-[10px] text-muted-foreground">
            Context: {contextUsage.percentage.toFixed(0)}% used
          </span>
          <span className="text-[10px] text-muted-foreground">
            ({contextUsage.remaining.toLocaleString()} tokens left)
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
            {model}
          </span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
            {tier}
          </span>
        </div>
      </div>

      {/* Main input area */}
      <div className="p-3">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message... (⌘+Enter to send)"
          className="w-full min-h-[80px] max-h-[200px] bg-transparent text-sm resize-none focus:outline-none placeholder:text-muted-foreground/40"
          rows={3}
        />
      </div>

      {/* Stats bar */}
      {showStats && value && (
        <div className="flex items-center justify-between px-3 py-1.5 border-t bg-muted/20">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Tokens:</span>
              <span className={cn(
                'text-[10px] font-medium',
                stats.tokens > 4000 ? 'text-destructive' :
                stats.tokens > 2000 ? 'text-warning' : 'text-foreground',
              )}>
                {stats.tokens.toLocaleString()}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Words:</span>
              <span className="text-[10px] font-medium">{stats.words.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Chars:</span>
              <span className="text-[10px] font-medium">{stats.chars.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-muted-foreground">Lines:</span>
              <span className="text-[10px] font-medium">{stats.lines}</span>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-muted-foreground">Est. cost:</span>
            <span className="text-[10px] font-medium text-primary">
              {formatCost(stats.estimatedCost)}
            </span>
          </div>
        </div>
      )}

      {/* Quick actions */}
      <div className="flex items-center justify-between px-3 py-2 border-t">
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="text-[10px] h-6"
            onClick={() => setShowActions(!showActions)}
          >
            {showActions ? 'Hide' : '⚡ Quick'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="text-[10px] h-6"
            onClick={() => setShowStats(!showStats)}
          >
            {showStats ? 'Hide Stats' : 'Stats'}
          </Button>
        </div>
        <div className="flex items-center gap-1 text-[10px] text-muted-foreground/40">
          <kbd className="rounded bg-muted/50 px-1 py-0.5 font-mono text-[9px]">⌘</kbd>
          <span>+</span>
          <kbd className="rounded bg-muted/50 px-1 py-0.5 font-mono text-[9px]">Enter</kbd>
          <span>send</span>
        </div>
      </div>

      {/* Quick actions panel */}
      {showActions && (
        <div className="px-3 pb-3 space-y-2 border-t">
          <div className="flex gap-1 pt-2 flex-wrap">
            <button
              type="button"
              onClick={() => setActionCategory(null)}
              className={cn(
                'text-[10px] px-2 py-0.5 rounded transition-colors',
                actionCategory === null
                  ? 'bg-primary/20 text-primary'
                  : 'text-muted-foreground hover:bg-muted/50',
              )}
            >
              All
            </button>
            {['prompt', 'format', 'context', 'tool'].map(cat => (
              <button
                key={cat}
                type="button"
                onClick={() => setActionCategory(cat)}
                className={cn(
                  'text-[10px] px-2 py-0.5 rounded capitalize transition-colors',
                  actionCategory === cat
                    ? 'bg-primary/20 text-primary'
                    : 'text-muted-foreground hover:bg-muted/50',
                )}
              >
                {cat}
              </button>
            ))}
          </div>
          <div className="grid grid-cols-5 gap-1">
            {filteredActions.map(action => (
              <button
                key={action.id}
                type="button"
                onClick={() => handleInsert(action.action)}
                className="flex flex-col items-center gap-0.5 p-1.5 rounded hover:bg-muted/50 transition-colors"
                title={action.label}
              >
                <span className="text-sm text-muted-foreground">{action.icon}</span>
                <span className="text-[9px] text-muted-foreground truncate w-full text-center">
                  {action.label}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
})