'use client'

import { useState, memo } from 'react'
import { useLocale } from '@/hooks/useLocale'
import { useRouter } from '@/vite/next-compat/navigation'
import { IconChat, IconEdit, IconBrain, IconBolt } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { FolderOpen, GraduationCap, LayoutDashboard, Lightbulb } from 'lucide-react'

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

// First-impression showcase: the journeys beyond this screen, mapped to the
// UX_FLOWS charter (Train My AI, Read My Files, Manage My Work).
const CAPABILITIES = [
  {
    title: 'Teach it',
    caption: 'Train on your data in 3 clicks',
    href: '/training',
    icon: GraduationCap,
  },
  {
    title: 'Bring your files',
    caption: 'Drop docs — it reads and answers',
    href: '/knowledge',
    icon: FolderOpen,
  },
  {
    title: 'Plan & track',
    caption: 'A kanban board that stays local',
    href: '/planner',
    icon: LayoutDashboard,
  },
]

function computeGreeting(): string {
  const h = new Date().getHours()
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}

export const EmptyState = memo(function EmptyState({
  hasModel,
  suggestions,
  onSuggestionClick,
}: EmptyStateProps) {
  const { t } = useLocale()
  const router = useRouter()
  // Lazy initializer: the greeting word is correct at first paint (the old
  // ''-then-effect version flashed "Ready!" before swapping to the real word).
  const [greeting] = useState(computeGreeting)

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
        <div className="absolute inset-0 rounded-2xl bg-primary/10 blur-2xl" />
        <div className="relative h-14 w-14 sm:h-16 sm:w-16 rounded-2xl bg-gradient-to-br from-primary/15 via-primary/10 to-accent/10 flex items-center justify-center border border-primary/10 shadow-sm">
          <IconChat className="h-6 w-6 sm:h-7 sm:w-7 text-primary/40" />
        </div>
      </div>

      {/* Greeting */}
      <div className="space-y-1.5">
        <p className="text-base font-semibold text-foreground tracking-tight">
          {hasModel ? greeting + '!' : t('common.starting')}
        </p>
        <p className="text-xs text-muted-foreground max-w-[280px] leading-relaxed">
          {hasModel
            ? 'Just ask — writing, translation, ideas, explanations, or a plain chat.'
            : t('common.starting_sub')}
        </p>
      </div>

      {/* Capability rail — what sloughGPT does beyond this chat */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 w-full max-w-lg">
        {CAPABILITIES.map((c) => {
          const Icon = c.icon
          return (
            <button
              key={c.href}
              type="button"
              onClick={() => router.push(c.href)}
              className="group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-inset flex flex-col items-center gap-1 px-3 py-3 rounded-xl border border-border/30 bg-card/30 hover:border-primary/20 hover:bg-primary/[0.03] hover:shadow-sm transition-all duration-200 cursor-pointer"
            >
              <Icon
                className="h-4 w-4 text-primary/60 group-hover:text-primary transition-colors"
                aria-hidden="true"
              />
              <span className="text-sm font-medium text-foreground">{c.title}</span>
              <span className="text-xs text-muted-foreground leading-relaxed">{c.caption}</span>
            </button>
          )
        })}
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
