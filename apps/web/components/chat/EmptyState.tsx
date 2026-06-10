'use client'

import { useEffect, useState } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { Chip } from '@/components/ui/tags'

interface EmptyStateProps {
  hasModel: boolean
  onSuggestionClick?: (text: string) => void
}

const moods = ['curious', 'friendly', 'playful', 'thoughtful', 'excited']
const moodEmojis = ['👋', '✨', '🤖', '💬', '🌟', '🚀', '🎯']

const suggestions = [
  { text: 'chat.suggestion.chat', icon: '💭' },
  { text: 'chat.suggestion.train', icon: '🧠' },
  { text: 'chat.suggestion.soul', icon: '🎭' },
  { text: 'chat.suggestion.models', icon: '📦' },
]

function MoodOrb() {
  const [pulse, setPulse] = useState(1)

  useEffect(() => {
    const interval = setInterval(() => {
      setPulse(p => 0.88 + Math.sin(Date.now() / 1400) * 0.12)
    }, 40)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="relative">
      <div
        className="h-16 w-16 sm:h-20 sm:w-20 rounded-full transition-transform duration-700 ease-in-out"
        style={{
          background: `radial-gradient(circle at 35% 30%, color-mix(in srgb, rgb(var(--primary)) 55%, transparent), color-mix(in srgb, rgb(var(--accent)) 30%, transparent))`,
          transform: `scale(${pulse})`,
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

export function EmptyState({ hasModel, onSuggestionClick }: EmptyStateProps) {
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

  return (
    <div className="flex flex-col items-center justify-center gap-5 py-8 sm:py-12 text-center px-4" role="region" aria-label="Chat ready">
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

      {hasModel && (
        <div className="w-full max-w-sm space-y-3 pt-2">
          <p className="text-[11px] text-muted-foreground/40 font-medium uppercase tracking-[0.1em]">Try asking</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {suggestions.map((s) => (
              <SuggestionChip
                key={s.text}
                text={t(s.text)}
                icon={s.icon}
                onClick={() => onSuggestionClick?.(t(s.text))}
              />
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 text-[11px] text-muted-foreground/40 bg-muted/20 px-3 py-1.5 rounded-full border border-border/30">
        <span className="flex items-center gap-1.5">
          <kbd className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[10px] shadow-sm border border-border/50">↵</kbd>
          <span>{t('chat.send')}</span>
        </span>
        <span className="text-muted-foreground/20">·</span>
        <span className="flex items-center gap-1.5">
          <kbd className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[10px] shadow-sm border border-border/50">⇧</kbd>
          <span>+</span>
          <kbd className="rounded-md bg-background px-1.5 py-0.5 font-mono text-[10px] shadow-sm border border-border/50">↵</kbd>
          <span>new line</span>
        </span>
      </div>
    </div>
  )
}
