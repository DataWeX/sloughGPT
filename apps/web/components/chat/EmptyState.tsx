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
  'chat.suggestion.chat',
  'chat.suggestion.train',
  'chat.suggestion.soul',
  'chat.suggestion.models',
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
        className="h-20 w-20 rounded-full transition-transform duration-700 ease-in-out"
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

function SuggestionChip({ text, onClick }: { text: string; onClick: () => void }) {
  return (
    <Chip
      label={text}
      onClick={onClick}
      className="w-full text-left px-4 py-2.5 text-xs rounded-lg"
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
    <div className="flex flex-col items-center justify-center gap-6 py-10 text-center px-4" role="region" aria-label="Chat ready">
      <MoodOrb />

      <div className="space-y-2">
        <p className="text-base font-semibold text-foreground">
          {hasModel ? greeting + '!' : t('common.starting')}
        </p>
        <p className="text-sm text-muted-foreground max-w-[260px]">
          {hasModel
            ? `I'm feeling ${moods[mood]} today ${moodEmojis[emoji]}`
            : t('common.starting_sub')}
        </p>
      </div>

      {hasModel && (
        <div className="w-full max-w-sm space-y-2">
          <p className="text-xs text-muted-foreground/50 font-medium uppercase tracking-wider">Try asking</p>
          <div className="grid grid-cols-1 gap-2">
            {suggestions.map((key) => (
              <SuggestionChip
                key={key}
                text={t(key)}
                onClick={() => onSuggestionClick?.(t(key))}
              />
            ))}
          </div>
        </div>
      )}

      <div className="flex items-center gap-3 text-xs text-muted-foreground/50 bg-muted/30 px-3 py-1.5 rounded-full">
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-background px-1.5 py-0.5 font-mono text-xs shadow-sm border">↵</kbd>
          <span>{t('chat.send')}</span>
        </span>
        <span className="text-muted-foreground/30">|</span>
        <span className="flex items-center gap-1">
          <kbd className="rounded bg-background px-1.5 py-0.5 font-mono text-xs shadow-sm border">Shift</kbd>
          <span>+ Enter for new line</span>
        </span>
      </div>
    </div>
  )
}
