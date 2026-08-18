'use client'

import { Card, CardHeader, CardTitle, CardContent } from '@sloughgpt/strui'
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
  if (!traits) return null
  const numericValues = (Object.values(traits) as unknown[]).filter((v): v is number => typeof v === 'number')
  if (numericValues.length === 0 || numericValues.every(v => v === 0)) return null

  const { dominant, weakest, spread } = traitBalance(traits)
  const type = personalityType(traits)
  const avg = numericValues.length > 0 ? numericValues.reduce((s, v) => s + v, 0) / numericValues.length : 0

  return (
    <Card data-testid="companion-insights">
      <CardHeader>
        <CardTitle className="text-base">Personality Profile</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Type</div>
            <div className="text-sm font-semibold">{type}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Dominant</div>
            <div className="text-sm font-semibold capitalize">{dominant}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Weakest</div>
            <div className="text-sm font-semibold capitalize">{weakest}</div>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Balance</div>
            <div className="text-sm font-semibold">
              {spread < 0.2 ? 'Even' : spread < 0.4 ? 'Moderate' : 'Skewed'}
            </div>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(Object.entries(traits) as [string, number][]).filter(([k]) => k !== 'name').map(([key, value]) => (
            <span
              key={key}
              className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                value >= 0.7 ? 'bg-primary/15 text-primary' :
                value >= 0.4 ? 'bg-muted text-muted-foreground' :
                'bg-muted/50 text-muted-foreground/60'
              }`}
            >
              {key} {(value * 100).toFixed(0)}%
            </span>
          ))}
        </div>
        {presets.length > 0 && (
          <div className="mt-3 text-[10px] text-muted-foreground">
            {presets.length} preset{presets.length !== 1 ? 's' : ''} available
          </div>
        )}
      </CardContent>
    </Card>
  )
}
