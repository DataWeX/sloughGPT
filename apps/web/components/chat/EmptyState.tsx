'use client'

import { useEffect, useState, memo } from 'react'
import Link from 'next/link'
import { useLocale } from '@/hooks/useLocale'
import { cn, Chip } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import type { Conversation } from '@/lib/session-controller'

interface EmptyStateProps {
  hasModel: boolean
  suggestions?: { text: string; icon: string }[]
  onSuggestionClick?: (text: string) => void
  recentConversations?: Conversation[]
  onLoadConversation?: (id: string) => void
}

const FALLBACK_SUGGESTIONS = [
  { text: 'Explain quantum computing in simple terms', icon: '🔬' },
  { text: 'Write a short poem about the ocean', icon: '🌊' },
  { text: 'Help me debug this Python code', icon: '🐍' },
  { text: 'What are the best practices for REST APIs?', icon: '💡' },
]
const moods = ['curious', 'friendly', 'playful', 'thoughtful', 'excited']
const moodEmojis = ['👋', '✨', '🤖', '💬', '🌟', '🚀', '🎯']

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
      className="w-full text-left px-4 py-2 sm:py-2.5 text-xs sm:text-sm rounded-lg border border-border/40 hover:border-primary/30 hover:bg-primary/5 transition-all cursor-pointer"
    />
  )
}

export const EmptyState = memo(function EmptyState({ hasModel, suggestions, onSuggestionClick, recentConversations, onLoadConversation }: EmptyStateProps) {
  const { t } = useLocale()
  const [greeting, setGreeting] = useState('')
  const [mood, setMood] = useState(0)
  const [emoji, setEmoji] = useState(0)

  useEffect(() => {
    const h = new Date().getHours()
    if (h < 12) setGreeting('Good morning')
    else if (h < 18) setGreeting('Good afternoon')
    else setGreeting('Good evening')

    const moodInterval = setInterval(() => {
      setMood(m => (m + 1) % moods.length)
      setEmoji(e => (e + 1) % moodEmojis.length)
    }, 4000)
    return () => clearInterval(moodInterval)
  }, [])

  const displaySuggestions = hasModel
    ? (suggestions && suggestions.length > 0 ? suggestions : FALLBACK_SUGGESTIONS)
    : null

  return (
    <div
      className="flex flex-col items-center justify-center gap-5 py-8 sm:py-12 text-center px-4"
      role="region"
      aria-label="Chat ready"
    >
      <MoodOrb />

      <div className="space-y-1.5">
        <p className="text-lg sm:text-xl font-semibold text-foreground tracking-tight">
          {hasModel ? (greeting || 'Ready') + '!' : t('common.starting')}
        </p>
        <p className="text-sm text-muted-foreground/80 max-w-[280px] leading-relaxed">
          {hasModel
            ? `I'm feeling ${moods[mood]} today ${moodEmojis[emoji]}`
            : t('common.starting_sub')}
        </p>
      </div>

      {displaySuggestions ? (
        <div className="w-full max-w-sm space-y-3 pt-2">
          <p className="text-[11px] text-muted-foreground/40 font-medium uppercase tracking-[0.1em]">Try asking</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
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
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"/></svg>
              Load a model to start chatting
            </Button>
          </Link>
        </div>
      ) : null}

      {recentConversations && recentConversations.length > 0 && onLoadConversation && (
        <div className="w-full max-w-sm space-y-2 pt-2">
          <p className="text-[11px] text-muted-foreground/40 font-medium uppercase tracking-[0.1em]">Recent</p>
          <div className="space-y-1">
            {recentConversations.slice(0, 4).map(conv => (
              <button
                key={conv.id}
                onClick={() => onLoadConversation(conv.id)}
                className="w-full text-left rounded-lg border border-border/40 px-3 py-2 hover:bg-muted/50 hover:border-primary/20 transition-all group"
              >
                <p className="text-xs font-medium truncate group-hover:text-primary transition-colors">{conv.name || 'Untitled'}</p>
                {conv.updated_at && (
                  <p className="text-[10px] text-muted-foreground/50 mt-0.5">
                    {new Date(conv.updated_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                  </p>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      <div
        className="flex flex-wrap items-center justify-center gap-3 text-[11px] text-muted-foreground/40 bg-muted/20 px-3 py-1.5 rounded-full border border-border/30"
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
