'use client'

import { cn, ActionCard, ChipGroup } from '@sloughgpt/strui'

interface SoulPersonalityCardProps {
  personality: Record<string, number> | undefined
  traits?: string[]
  soulName?: string
}

function traitLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
}

function traitColor(value: number): string {
  if (value >= 0.8) return 'bg-success'
  if (value >= 0.6) return 'bg-primary'
  if (value >= 0.4) return 'bg-warning'
  return 'bg-muted-foreground/40'
}

export function SoulPersonalityCard({ personality, traits, soulName }: SoulPersonalityCardProps) {
  if (!personality || Object.keys(personality).length === 0) return null

  const entries = Object.entries(personality).sort((a, b) => b[1] - a[1])
  const avgScore = entries.reduce((s, [, v]) => s + v, 0) / entries.length

  return (
    <ActionCard
      title={`Personality${soulName ? ` — ${soulName}` : ''}`}
      subtitle={`avg ${(avgScore * 100).toFixed(0)}%`}
      testId="soul-personality"
    >
      {traits && traits.length > 0 && (
        <div className="mb-3">
          <ChipGroup
            chips={traits.map(t => ({ children: t, className: 'bg-primary/10 text-primary' }))}
          />
        </div>
      )}
      <div className="space-y-2">
        {entries.map(([key, value]) => (
          <div key={key}>
            <div className="flex items-center justify-between text-[11px] mb-0.5">
              <span className="text-muted-foreground">{traitLabel(key)}</span>
              <span className="font-mono">{(value * 100).toFixed(0)}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-muted/50 overflow-hidden">
              <div
                className={cn('h-full rounded-full transition-all', traitColor(value))}
                style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </ActionCard>
  )
}
