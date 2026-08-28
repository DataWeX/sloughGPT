'use client'

import { useEffect, useState, memo } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { cn, IconChat, IconBolt } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Atom, Waves, Bug, Lightbulb } from 'lucide-react'

interface EmptyStateProps {
  hasModel: boolean
  suggestions?: { text: string; icon: React.ReactNode }[]
  onSuggestionClick?: (text: string) => void
}

const FALLBACK_SUGGESTIONS: { text: string; icon: React.ReactNode }[] = [
  { text: 'Explain quantum computing in simple terms', icon: <Atom className="h-3.5 w-3.5" /> },
  { text: 'Write a short poem about the ocean', icon: <Waves className="h-3.5 w-3.5" /> },
  { text: 'Help me debug this Python code', icon: <Bug className="h-3.5 w-3.5" /> },
  { text: 'What are the best practices for REST APIs?', icon: <Lightbulb className="h-3.5 w-3.5" /> },
]

interface SuggestionChipProps {
  text: string
  icon: React.ReactNode
  onClick: () => void
}

function SuggestionChip({ text, icon, onClick }: SuggestionChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="w-full text-left px-3 py-2 text-xs rounded-lg border border-border/50 bg-card hover:border-primary/30 hover:bg-primary/5 transition-all cursor-pointer flex items-center gap-2"
    >
      <span className="text-muted-foreground/60 shrink-0">{icon}</span>
      <span>{text}</span>
    </button>
  )
}

export const EmptyState = memo(function EmptyState({ hasModel, suggestions, onSuggestionClick }: EmptyStateProps) {
  const { t } = useLocale()
  const [greeting, setGreeting] = useState('')

  useEffect(() => {
    const h = new Date().getHours()
    if (h < 12) setGreeting('Good morning')
    else if (h < 18) setGreeting('Good afternoon')
    else setGreeting('Good evening')
  }, [])

  const displaySuggestions = hasModel
    ? (suggestions && suggestions.length > 0 ? suggestions : FALLBACK_SUGGESTIONS)
    : null

  return (
    <div
      className="flex flex-col items-center justify-center gap-4 py-8 sm:py-12 text-center px-4"
      role="region"
      aria-label="Chat ready"
    >
      {/* Protocol identity mark */}
      <div className="relative" aria-hidden="true">
        <div
          className="h-12 w-12 sm:h-14 sm:w-14 rounded-xl bg-gradient-to-br from-primary/20 via-primary/10 to-accent/15 flex items-center justify-center border border-primary/10"
        >
          <IconChat className="h-5 w-5 sm:h-6 sm:w-6 text-primary/60" />
        </div>
      </div>

      <div className="space-y-1.5">
        <p className="text-base font-semibold text-foreground tracking-tight">
          {hasModel ? (greeting || 'Ready') + '!' : t('common.starting')}
        </p>
        <p className="text-xs text-muted-foreground/70 max-w-[260px] leading-relaxed">
          {hasModel
            ? 'Ask me anything — I\'m here to help.'
            : t('common.starting_sub')}
        </p>
      </div>

      {displaySuggestions ? (
        <div className="w-full max-w-sm space-y-2 pt-1">
          <p className="text-[10px] text-muted-foreground/40 font-medium uppercase tracking-[0.1em]">Try asking</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
            {displaySuggestions.map((s) => (
              <SuggestionChip
                key={s.text}
                text={s.text}
                icon={s.icon}
                onClick={() => onSuggestionClick?.(s.text)}
              />
            ))}
          </div>
        </div>
      ) : null}

      {!hasModel && (
        <div className="pt-1">
          <Button size="sm" className="gap-1.5 h-8 text-xs">
            <IconBolt className="w-3 h-3" aria-hidden="true" />
            Load a model to start
          </Button>
        </div>
      )}

      <div
        className="flex flex-wrap items-center justify-center gap-2.5 text-[10px] text-muted-foreground/35"
        aria-label="Keyboard shortcuts"
      >
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-muted/60 px-1 py-0.5 font-mono text-[9px] border border-border/40">↵</kbd>
          <span>{t('chat.send')}</span>
        </span>
        <span className="text-muted-foreground/15">·</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-muted/60 px-1 py-0.5 font-mono text-[9px] border border-border/40">/</kbd>
          <span>commands</span>
        </span>
        <span className="text-muted-foreground/15">·</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-muted/60 px-1 py-0.5 font-mono text-[9px] border border-border/40">?</kbd>
          <span>shortcuts</span>
        </span>
      </div>
    </div>
  )
})
