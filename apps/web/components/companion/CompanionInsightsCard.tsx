'use client'

import { useMemo } from 'react'
import { cn, InsightsCard, ChipGroup } from '@sloughgpt/strui'
import type { CompanionTraits } from '@/lib/companion-controller'

interface CompanionInsightsCardProps {
  traits: CompanionTraits | null
  presets: Array<{ id: string; name: string; description?: string }>
}

function traitBalance(traits: CompanionTraits): { dominant: string; weakest: string; spread: number } {
  const entries = (Object.entries(traits) as [string, number][]).filter(([k]) => k !== 'id' && k !== 'name')
  if (entries.length === 0) return { dominant: '—', weakest: '—', spread: 0 }
  entries.sort((a, b) => b[1] - a[1])
  return {
    dominant: entries[0][0],
    weakest: entries[entries.length - 1][0],
    spread: entries[0][1] - entries[entries.length - 1][1],
  }
}

function personalityType(traits: CompanionTraits): string {
  const numericValues = (Object.values(traits) as unknown[]).filter((v): v is number => typeof v === 'number')
  const avg = numericValues.reduce((s, v) => s + v, 0) / numericValues.length
  const entries = (Object.entries(traits) as [string, number][]).filter(([k]) => k !== 'name')
  const maxTrait = entries.sort((a, b) => b[1] - a[1])[0]

  if (maxTrait[1] >= 0.8) return `Strong ${maxTrait[0]}`
  if (avg >= 0.7) return 'Balanced'
  if (avg <= 0.3) return 'Reserved'
  return 'Moderate'
}

export function CompanionInsightsCard({ traits, presets }: CompanionInsightsCardProps) {
  const numericValues = useMemo(() => (Object.values(traits ?? {}) as unknown[]).filter((v): v is number => typeof v === 'number'), [traits])
  const { dominant, weakest, spread } = useMemo(() => traitBalance(traits!), [traits])
  const type = useMemo(() => personalityType(traits!), [traits])

  if (!traits) return null
  if (numericValues.length === 0 || numericValues.every(v => v === 0)) return null

  const balanceLabel = spread < 0.2 ? 'Even' : spread < 0.4 ? 'Moderate' : 'Skewed'

  const traitEntries = (Object.entries(traits) as [string, number][]).filter(([k]) => k !== 'name')

  return (
    <InsightsCard
      title="Personality Profile"
      testId="companion-insights"
      kpis={[
        { label: 'Type', value: type },
        { label: 'Dominant', value: <span className="capitalize">{dominant}</span> },
        { label: 'Weakest', value: <span className="capitalize">{weakest}</span> },
        { label: 'Balance', value: balanceLabel },
      ]}
      kpiColumns={4}
    >
      <ChipGroup
        chips={traitEntries.map(([key, value]) => ({
          children: `${key} ${(value * 100).toFixed(0)}%`,
          className: cn(value >= 0.7 ? 'bg-primary/15 text-primary' :
            value >= 0.4 ? 'bg-muted text-muted-foreground' :
            'bg-muted/50 text-muted-foreground/60'),
        }))}
      />
      {presets.length > 0 && (
        <div className="mt-2 text-[10px] text-muted-foreground">
          {presets.length} preset{presets.length !== 1 ? 's' : ''} available
        </div>
      )}
    </InsightsCard>
  )
}
