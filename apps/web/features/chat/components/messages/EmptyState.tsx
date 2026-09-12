'use client'

import { useEffect, useState, memo, useCallback } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { IconChat, IconEdit, IconBrain, IconBolt, IconSearch, IconVision, IconSparkle, IconDocument, cn } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Waves, Lightbulb, Mic } from 'lucide-react'

interface EmptyStateProps {
  hasModel: boolean
  suggestions?: { text: string; icon: React.ReactNode }[]
  onSuggestionClick?: (text: string) => void
  onModeSelect?: (mode: string) => void
}

interface ToolModeCardProps {
  mode: string
  label: string
  description: string
  icon: React.ReactNode
  color: string
  onClick: () => void
}

function ToolModeCard({ mode, label, description, icon, color, onClick }: ToolModeCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "group relative flex flex-col items-start gap-1.5 p-3 rounded-xl border border-border/30 bg-card/30",
        "hover:border-primary/30 hover:bg-primary/[0.03] hover:shadow-sm hover:shadow-primary/5",
        "hover:-translate-y-0.5 active:scale-[0.98] transition-all duration-200 cursor-pointer text-left"
      )}
    >
      <div className="flex items-center gap-2">
        <div className={cn("flex items-center justify-center h-7 w-7 rounded-lg transition-colors duration-200", color)}>
          {icon}
        </div>
        <span className="text-xs font-medium text-foreground/90 group-hover:text-foreground transition-colors">
          {label}
        </span>
      </div>
      <p className="text-[11px] text-muted-foreground/50 leading-snug group-hover:text-muted-foreground/70 transition-colors">
        {description}
      </p>
    </button>
  )
}

const TOOL_MODES = [
  { mode: 'write', label: 'Write', description: 'Emails, stories, posts', icon: <IconEdit className="h-3.5 w-3.5" />, color: 'bg-violet-500/10 text-violet-500/70' },
  { mode: 'translate', label: 'Translate', description: 'Across languages', icon: <IconVision className="h-3.5 w-3.5" />, color: 'bg-blue-500/10 text-blue-500/70' },
  { mode: 'rewrite', label: 'Rewrite', description: 'Polish your text', icon: <IconSparkle className="h-3.5 w-3.5" />, color: 'bg-amber-500/10 text-amber-500/70' },
  { mode: 'brainstorm', label: 'Brainstorm', description: 'Generate ideas', icon: <IconBolt className="h-3.5 w-3.5" />, color: 'bg-emerald-500/10 text-emerald-500/70' },
  { mode: 'decide', label: 'Decide', description: 'Compare options', icon: <IconBrain className="h-3.5 w-3.5" />, color: 'bg-rose-500/10 text-rose-500/70' },
  { mode: 'explain', label: 'Explain', description: 'Simplify topics', icon: <IconSearch className="h-3.5 w-3.5" />, color: 'bg-cyan-500/10 text-cyan-500/70' },
  { mode: 'wellness', label: 'Wellness', description: 'Calm your mind', icon: <Waves className="h-3.5 w-3.5" />, color: 'bg-teal-500/10 text-teal-500/70' },
]

const QUICK_STARTERS = [
  { text: 'Explain quantum computing simply', icon: <Lightbulb className="h-3.5 w-3.5" /> },
  { text: 'Write a professional email', icon: <IconEdit className="h-3.5 w-3.5" /> },
  { text: 'Help me decide between two options', icon: <IconBrain className="h-3.5 w-3.5" /> },
  { text: 'Brainstorm weekend activities', icon: <IconBolt className="h-3.5 w-3.5" /> },
]

export const EmptyState = memo(function EmptyState({ hasModel, suggestions, onSuggestionClick, onModeSelect }: EmptyStateProps) {
  const { t } = useLocale()
  const [greeting, setGreeting] = useState('')

  useEffect(() => {
    const h = new Date().getHours()
    if (h < 12) setGreeting('Good morning')
    else if (h < 18) setGreeting('Good afternoon')
    else setGreeting('Good evening')
  }, [])

  const handleModeClick = useCallback((mode: string) => {
    onModeSelect?.(mode)
  }, [onModeSelect])

  const displayStarters = hasModel
    ? (suggestions && suggestions.length > 0 ? suggestions : QUICK_STARTERS)
    : null

  return (
    <div
      className="flex flex-col items-center justify-center gap-6 py-10 sm:py-14 text-center px-4 animate-in fade-in duration-500"
      role="region"
      aria-label="Chat ready"
    >
      {/* Hero */}
      <div className="relative" aria-hidden="true">
        <div className="absolute inset-0 rounded-2xl bg-primary/8 blur-2xl" />
        <div className="relative h-14 w-14 sm:h-16 sm:w-16 rounded-2xl bg-gradient-to-br from-primary/15 via-primary/8 to-accent/10 flex items-center justify-center border border-primary/10 shadow-sm">
          <IconChat className="h-6 w-6 sm:h-7 sm:w-7 text-primary/40" />
        </div>
      </div>

      {/* Greeting */}
      <div className="space-y-1.5">
        <p className="text-lg font-semibold text-foreground tracking-tight">
          {hasModel ? (greeting || 'Ready') + '!' : t('common.starting')}
        </p>
        <p className="text-xs text-muted-foreground/50 max-w-[280px] leading-relaxed">
          {hasModel
            ? 'Chat, write, translate, or brainstorm — pick a tool or just ask.'
            : t('common.starting_sub')}
        </p>
      </div>

      {/* Tool Mode Cards */}
      {hasModel && (
        <div className="w-full max-w-lg space-y-3">
          <p className="text-[10px] text-muted-foreground/30 font-medium uppercase tracking-[0.12em]">Quick tools</p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {TOOL_MODES.map((tool) => (
              <ToolModeCard
                key={tool.mode}
                mode={tool.mode}
                label={tool.label}
                description={tool.description}
                icon={tool.icon}
                color={tool.color}
                onClick={() => handleModeClick(tool.mode)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Quick Starters */}
      {displayStarters && (
        <div className="w-full max-w-lg space-y-2.5 pt-1">
          <p className="text-[10px] text-muted-foreground/30 font-medium uppercase tracking-[0.12em]">Or try asking</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {displayStarters.map((s) => (
              <button
                key={s.text}
                type="button"
                onClick={() => onSuggestionClick?.(s.text)}
                className="w-full text-left px-3 py-2 text-xs rounded-xl border border-border/30 bg-card/30 hover:border-primary/20 hover:bg-primary/[0.03] hover:shadow-sm transition-all duration-200 cursor-pointer flex items-center gap-2 group"
              >
                <span className="text-muted-foreground/30 group-hover:text-primary/50 shrink-0 transition-colors">{s.icon}</span>
                <span className="text-muted-foreground/60 group-hover:text-foreground/80 transition-colors truncate">{s.text}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {!hasModel && (
        <div className="pt-1">
          <Button size="sm" className="gap-1.5 h-8 text-xs">
            <IconBolt className="w-3 h-3" aria-hidden="true" />
            Load a model to start
          </Button>
        </div>
      )}

      {/* Keyboard hints */}
      <div
        className="flex flex-wrap items-center justify-center gap-2.5 text-[10px] text-muted-foreground/25"
        aria-label="Keyboard shortcuts"
      >
        <span className="flex items-center gap-1">
          <kbd className="rounded-md bg-muted/40 px-1.5 py-0.5 font-mono text-[9px] border border-border/20">↵</kbd>
          <span>{t('chat.send')}</span>
        </span>
        <span className="text-muted-foreground/10">·</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded-md bg-muted/40 px-1.5 py-0.5 font-mono text-[9px] border border-border/20">/</kbd>
          <span>commands</span>
        </span>
        <span className="text-muted-foreground/10">·</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded-md bg-muted/40 px-1.5 py-0.5 font-mono text-[9px] border border-border/20">?</kbd>
          <span>shortcuts</span>
        </span>
      </div>
    </div>
  )
})
