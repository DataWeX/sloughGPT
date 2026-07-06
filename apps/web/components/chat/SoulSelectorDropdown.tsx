'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'

import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { IconChevronDown, IconCheck, IconHeart, IconChevronRight } from '@sloughgpt/strui'
import type { Soul } from '@/lib/souls-controller'
import { soulsController } from '@/lib/souls-controller'
import { deriveArchetype } from '@/components/souls/PersonalitySummary'
import { useChatToolbarContext } from '@/contexts/ChatToolbarContext'

export function SoulSelectorDropdown() {
  const ctx = useChatToolbarContext()
  const { souls, current: currentSoul, onSelect } = ctx.soul

  const GROUP_COLORS: Record<string, string> = {
    personality: '#8b5cf6',
    cognition: '#3b82f6',
    emotion: '#ec4899',
  }
  const router = useRouter()
  const [traitWeights, setTraitWeights] = useState<Record<string, Record<string, number>> | null>(null)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    soulsController.getTraitWeights()
      .then(w => {
        if (!cancelled && w && !('error' in w)) {
          setTraitWeights(w as Record<string, Record<string, number>>)
        }
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [open])

  const archetype = traitWeights ? deriveArchetype(traitWeights) : null

  const groupAverages: { label: string; value: number; color: string }[] = traitWeights
    ? (['personality', 'cognition', 'emotion'] as const).map(g => {
        const traits = traitWeights[g]
        if (!traits) return { label: g, value: 0, color: GROUP_COLORS[g] }
        const entries = Object.entries(traits).filter(([, v]) => typeof v === 'number')
        const avg = entries.length > 0
          ? Math.round(entries.reduce((s, [, v]) => s + v * 100, 0) / entries.length)
          : 0
        return { label: g.charAt(0).toUpperCase() + g.slice(1), value: avg, color: GROUP_COLORS[g] }
      })
    : []

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-1.5 sm:px-2.5 text-xs gap-1 sm:gap-1.5 rounded-full bg-primary/8 text-primary hover:bg-primary/15 border border-primary/15" title={currentSoul?.traits?.join(', ')}>
          <IconHeart className="h-3 w-3 shrink-0" />
          <span className="hidden sm:inline truncate max-w-[64px]">{currentSoul?.name || 'Personality'}</span>
          <IconChevronDown className="hidden sm:block h-2.5 w-2.5 opacity-40 shrink-0" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-64 p-0">
        {/* ── Profile card ── */}
        <div className="px-3 pt-3 pb-2">
          <div className="flex items-center justify-between mb-1">
            <span className="text-sm font-semibold capitalize">{currentSoul?.name || 'Default'}</span>
            {archetype && (
              <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-primary/10 text-primary">
                {archetype.label}
              </span>
            )}
          </div>
          {archetype && (
            <p className="text-[10px] text-muted-foreground leading-relaxed mb-2">{archetype.description}</p>
          )}

          {/* Group averages mini bars */}
          {groupAverages.length > 0 && (
            <div className="space-y-1 mb-2">
              {groupAverages.map(g => (
                <div key={g.label} className="flex items-center gap-2">
                  <span className="text-[9px] text-muted-foreground w-16 shrink-0">{g.label}</span>
                  <div className="flex-1 h-1 rounded-full bg-muted-foreground/10 overflow-hidden">
                    <div className="h-full rounded-full transition-all" style={{ width: `${g.value}%`, backgroundColor: g.color, opacity: 0.6 }} />
                  </div>
                  <span className="text-[9px] font-mono tabular-nums w-5 text-right" style={{ color: g.color }}>{g.value}</span>
                </div>
              ))}
            </div>
          )}

          <button
            type="button"
            onClick={() => { setOpen(false); router.push('/models') }}
            className="inline-flex items-center gap-1 text-[10px] text-primary hover:text-primary/80 transition-colors"
          >
            View full profile
            <IconChevronRight className="h-2.5 w-2.5" />
          </button>
        </div>

        <div className="border-t border-border/40 my-0" />

        {/* ── Soul list ── */}
        <div className="py-1 max-h-48 overflow-y-auto">
          {souls.map((s) => (
            <DropdownMenuItem
              key={s.name}
              onSelect={() => onSelect(s)}
              className="justify-between text-xs px-3 py-1.5"
            >
              <div className="flex items-center gap-2">
                <IconHeart className="h-3 w-3 text-muted-foreground shrink-0" />
                <span className="capitalize">{s.name}</span>
              </div>
              {currentSoul?.name === s.name && <IconCheck className="h-3 w-3 text-primary shrink-0" />}
            </DropdownMenuItem>
          ))}
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
