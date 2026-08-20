'use client'

import { useEffect, useState, memo } from 'react'
import Link from 'next/link'
import { useLocale } from '@/hooks/useLocale'
import { cn, Chip, IconBeaker } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'

interface EmptyStateProps {
  hasModel: boolean
  suggestions?: { text: string; icon: string }[]
  onSuggestionClick?: (text: string) => void
}

const FALLBACK_SUGGESTIONS = [
  { text: 'Explain quantum computing in simple terms', icon: '🔬' },
  { text: 'Write a short poem about the ocean', icon: '🌊' },
  { text: 'Help me debug this Python code', icon: '🐍' },
  { text: 'What are the best practices for REST APIs?', icon: '💡' },
]

function MoodOrb() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false)

  useEffect(() => {
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)')
    setPrefersReducedMotion(mql.matches)
    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches)
    mql.addEventListener('change', handler)
    return () => mql.removeEventListener('change', handler)
  }, [])

  return (
    <div className="relative" aria-hidden="true">
      <div
        className={cn(
          "h-16 w-16 sm:h-20 sm:w-20 rounded-full",
          !prefersReducedMotion && "mood-orb-pulse"
        )}
        style={{
          background: `radial-gradient(circle at 35% 30%, color-mix(in srgb, rgb(var(--primary)) 55%, transparent), color-mix(in srgb, rgb(var(--accent)) 30%, transparent))`,
          boxShadow: `0 0 40px color-mix(in srgb, rgb(var(--primary)) 20%, transparent), 0 0 80px color-mix(in srgb, rgb(var(--accent)) 12%, transparent)`,
        }}
      />
      <div className="absolute -inset-4 rounded-full opacity-25 blur-xl"
        style={{
          background: `radial-gradient(circle, color-mix(in srgb, rgb(var(--primary)) 25%, transparent), transparent 70%)`,
        }}
      />
    </div>
  )
}

function SuggestionChip({ text, icon, onClick }: { text: string; icon: string; onClick: () => void }) {
  return (
    <Chip
      label={`${icon}  ${text}`}
      onClick={onClick}
      className="w-full text-left px-4 py-2.5 text-xs sm:text-sm rounded-xl border border-border/50 bg-card hover:border-primary/30 hover:bg-primary/5 transition-all cursor-pointer shadow-sm"
    />
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
      className="flex flex-col items-center justify-center gap-6 py-10 sm:py-16 text-center px-4"
      role="region"
      aria-label="Chat ready"
    >
      <MoodOrb />

      <div className="space-y-2">
        <p className="text-lg sm:text-xl font-semibold text-foreground tracking-tight">
          {hasModel ? (greeting || 'Ready') + '!' : t('common.starting')}
        </p>
        <p className="text-sm text-muted-foreground/70 max-w-[300px] leading-relaxed">
          {hasModel
            ? 'Ask me anything — I\'m here to help.'
            : t('common.starting_sub')}
        </p>
      </div>

      {displaySuggestions ? (
        <div className="w-full max-w-sm space-y-4 pt-2">
          <p className="text-[11px] text-muted-foreground/50 font-medium uppercase tracking-[0.1em]">Try asking</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
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

      {!hasModel ? (
        <div className="pt-2">
          <Link href="/models">
            <Button size="sm" className="gap-1.5">
              <IconBeaker className="w-3.5 h-3.5" />
              Load a model to start chatting
            </Button>
          </Link>
        </div>
      ) : (
        <div className="flex items-center gap-2 pt-1">
          <Button size="sm" variant="outline" className="h-8 text-xs" onClick={() => onSuggestionClick?.('Show me my datasets')}>
            Datasets
          </Button>
          <Button size="sm" className="h-8 text-xs" onClick={() => onSuggestionClick?.('Train a model on my data')}>
            Train model
          </Button>
        </div>
      )}

      <div
        className="flex flex-wrap items-center justify-center gap-3 text-[11px] text-muted-foreground/40 bg-muted/30 px-4 py-2 rounded-full border border-border/30"
        aria-label="Keyboard shortcuts"
      >
        <span className="flex items-center gap-1.5">
          <kbd className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[10px] shadow-sm border border-border/50">↵</kbd>
          <span>{t('chat.send')}</span>
        </span>
        <span className="text-muted-foreground/20">·</span>
        <span className="flex items-center gap-1.5">
          <kbd className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[10px] shadow-sm border border-border/50">/</kbd>
          <span>commands</span>
        </span>
        <span className="text-muted-foreground/20">·</span>
        <span className="flex items-center gap-1.5">
          <kbd className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[10px] shadow-sm border border-border/50">?</kbd>
          <span>shortcuts</span>
        </span>
      </div>
    </div>
  )
})
