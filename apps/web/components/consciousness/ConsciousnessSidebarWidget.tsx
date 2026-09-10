'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { cn } from '@sloughgpt/strui'
import { IconBrain } from '@/components/icons/NavIcons'
import { useConsciousnessStatus, getQualiaMood, getConsciousnessLevelLabel } from '@/hooks/useConsciousnessStatus'
import { useLocale } from '@/hooks/useLocale'

export function ConsciousnessSidebarWidget() {
  const { status } = useConsciousnessStatus()
  const { t } = useLocale()
  const router = useRouter()
  const [expanded, setExpanded] = useState(false)
  const panelRef = useRef<HTMLDivElement>(null)

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
      setExpanded(false)
    }
  }, [])

  useEffect(() => {
    if (!expanded) return
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [expanded, handleClickOutside])

  const enabled = status?.enabled ?? false
  const level = status?.level ?? 0
  const episodes = status?.episodes ?? 0
  const qualia = status?.current_qualia
  const mood = getQualiaMood(qualia)

  const valence = qualia?.valence ?? 0
  const arousal = qualia?.arousal ?? 0
  const novelty = qualia?.novelty ?? 0

  return (
    <div className="relative" ref={panelRef}>
      <button
        onClick={() => setExpanded(prev => !prev)}
        className="group relative flex min-h-10 w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[11px] transition-colors duration-200 ease-smooth text-foreground/78 hover:bg-primary/10 hover:text-primary dark:text-muted-foreground"
        aria-expanded={expanded}
        aria-label={t('consciousness_sidebar.aria_label')}
      >
        <IconBrain className="h-4 w-4 shrink-0 opacity-90 dark:opacity-80" aria-hidden />
        <span className="flex-1 truncate">{t('consciousness_sidebar.level_badge', { level })}</span>
        {mood && <span className="truncate text-muted-foreground/60">{mood}</span>}
        <span className="relative flex h-2 w-2 shrink-0">
          <span className={cn('absolute inline-flex h-full w-full animate-ping rounded-full opacity-75', enabled ? 'bg-green-400' : 'bg-gray-400')} />
          <span className={cn('relative inline-flex h-2 w-2 rounded-full', enabled ? 'bg-green-500' : 'bg-gray-500')} />
        </span>
      </button>

      {expanded && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 rounded-lg border border-border/40 bg-popover p-3 shadow-xl">
          <div className="mb-2 space-y-1.5">
            <MiniQualiaBar label={t('consciousness_sidebar.qualia_valence')} value={valence} color="bg-green-500" />
            <MiniQualiaBar label={t('consciousness_sidebar.qualia_arousal')} value={arousal} color="bg-amber-500" />
            <MiniQualiaBar label={t('consciousness_sidebar.qualia_novelty')} value={novelty} color="bg-violet-500" />
          </div>

          <div className="mb-2 flex items-center justify-between text-[10px] text-muted-foreground/70">
            <span>{t('consciousness_sidebar.episodes', { count: episodes })}</span>
            <span>{getConsciousnessLevelLabel(level)}</span>
          </div>

          <div className="border-t border-border/30 pt-2">
            <QuickLink
              label={t('consciousness_sidebar.link_dashboard')}
              onClick={() => { router.push('/consciousness/dashboard'); setExpanded(false) }}
            />
            <QuickLink
              label={t('consciousness_sidebar.link_settings')}
              onClick={() => { router.push('/consciousness/settings'); setExpanded(false) }}
            />
            <QuickLink
              label={t('consciousness_sidebar.link_chat')}
              onClick={() => { router.push('/chat'); setExpanded(false) }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

function MiniQualiaBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(((value + 1) / 2) * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="w-12 truncate text-[10px] text-muted-foreground/60">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
        <div className={cn('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-6 text-right text-[10px] tabular-nums text-muted-foreground/60">{pct}</span>
    </div>
  )
}

function QuickLink({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="w-full rounded px-2 py-1 text-left text-[11px] text-foreground/78 transition-colors hover:bg-primary/10 hover:text-primary"
    >
      {label}
    </button>
  )
}
