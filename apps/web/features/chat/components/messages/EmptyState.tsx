'use client'

import { useEffect, useState, memo } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { IconChat, IconEdit, IconBrain, IconBolt, cn } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Lightbulb } from 'lucide-react'

interface EmptyStateProps {
  hasModel: boolean
  suggestions?: { text: string; icon: React.ReactNode }[]
  onSuggestionClick?: (text: string) => void
}

const QUICK_STARTERS = [
  { text: 'Explain quantum computing simply', icon: <Lightbulb className="h-3.5 w-3.5" /> },
  { text: 'Write a professional email', icon: <IconEdit className="h-3.5 w-3.5" /> },
  { text: 'Help me decide between two options', icon: <IconBrain className="h-3.5 w-3.5" /> },
  { text: 'Brainstorm weekend activities', icon: <IconBolt className="h-3.5 w-3.5" /> },
]

export const EmptyState = memo(function EmptyState({
  hasModel,
  suggestions,
  onSuggestionClick,
}: EmptyStateProps) {
  const { t } = useLocale()
  const [greeting, setGreeting] = useState('')

  useEffect(() => {
    const h = new Date().getHours()
    if (h < 12) setGreeting('Good morning')
    else if (h < 18) setGreeting('Good afternoon')
    else setGreeting('Good evening')
  }, [])

  const displayStarters = hasModel
    ? suggestions && suggestions.length > 0
      ? suggestions
      : QUICK_STARTERS
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
        <p className="text-base font-semibold text-foreground tracking-tight">
          {hasModel ? (greeting || 'Ready') + '!' : t('common.starting')}
        </p>
        <p className="text-xs text-muted-foreground max-w-[280px] leading-relaxed">
          {hasModel
            ? 'Just ask — writing, translation, ideas, explanations, or a plain chat.'
            : t('common.starting_sub')}
        </p>
      </div>

      {/* Quick Starters */}
      {displayStarters && (
        <div className="w-full max-w-lg space-y-2.5 pt-1">
          <p className="text-xs text-muted-foreground font-medium uppercase tracking-[0.12em]">
            Or try asking
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {displayStarters.map((s) => (
              <button
                key={s.text}
                type="button"
                onClick={() => onSuggestionClick?.(s.text)}
                className="focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset w-full text-left px-3 py-2 text-xs rounded-xl border border-border/30 bg-card/30 hover:border-primary/20 hover:bg-primary/[0.03] hover:shadow-sm transition-all duration-200 cursor-pointer flex items-center gap-2 group"
              >
                <span className="text-muted-foreground group-hover:text-primary/50 shrink-0 transition-colors">
                  {s.icon}
                </span>
                <span className="text-muted-foreground group-hover:text-foreground/80 transition-colors truncate">
                  {s.text}
                </span>
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
        className="flex flex-wrap items-center justify-center gap-2.5 text-xs text-muted-foreground"
        aria-label="Keyboard shortcuts"
      >
        <span className="flex items-center gap-1">
          <kbd className="rounded-md bg-muted/40 px-1.5 py-0.5 font-mono text-[11px] border border-border/20">
            ↵
          </kbd>
          <span>{t('chat.send')}</span>
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded-md bg-muted/40 px-1.5 py-0.5 font-mono text-[11px] border border-border/20">
            /
          </kbd>
          <span>commands</span>
        </span>
        <span className="text-muted-foreground">·</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded-md bg-muted/40 px-1.5 py-0.5 font-mono text-[11px] border border-border/20">
            ?
          </kbd>
          <span>shortcuts</span>
        </span>
      </div>
    </div>
  )
})
