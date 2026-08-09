'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'

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
    <Card data-testid="soul-personality">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Personality{soulName ? ` — ${soulName}` : ''}</CardTitle>
          <span className="text-[10px] font-mono text-muted-foreground">
            avg {(avgScore * 100).toFixed(0)}%
          </span>
        </div>
      </CardHeader>
      <CardContent>
        {traits && traits.length > 0 && (
          <div className="flex flex-wrap gap-1 mb-3">
            {traits.map(t => (
              <span key={t} className="text-[9px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                {t}
              </span>
            ))}
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
                  className={`h-full rounded-full transition-all ${traitColor(value)}`}
                  style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
