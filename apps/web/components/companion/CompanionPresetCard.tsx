'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

interface CompanionPreset {
  id: string
  name: string
  description: string
}

interface CompanionPresetCardProps {
  presets: CompanionPreset[]
  activePreset?: string
  onSelect?: (presetId: string) => void
}

const PRESET_COLORS: Record<string, string> = {
  warm: 'border-warning/50 bg-warning/5',
  curious: 'border-primary/50 bg-primary/5',
  creative: 'border-accent/50 bg-accent/5',
  confident: 'border-success/50 bg-success/5',
  professional: 'border-muted-foreground/50 bg-muted/5',
  playful: 'border-destructive/50 bg-destructive/5',
}

export function CompanionPresetCard({ presets, activePreset, onSelect }: CompanionPresetCardProps) {
  const [selected, setSelected] = useState<string | null>(activePreset ?? null)

  if (presets.length === 0) return null

  return (
    <Card data-testid="companion-preset">
      <CardHeader>
        <CardTitle className="text-base">Presets</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {presets.map(p => {
            const isActive = selected === p.id
            return (
              <button
                key={p.id}
                className={cn(
                  'text-left p-2.5 rounded-lg border transition-all',
                  isActive
                    ? (PRESET_COLORS[p.id] ?? 'border-primary/50 bg-primary/5')
                    : 'border-border hover:border-primary/30'
                )}
                onClick={() => {
                  setSelected(p.id)
                  onSelect?.(p.id)
                }}
                data-testid={`preset-${p.id}`}
              >
                <div className="text-xs font-medium capitalize">{p.name}</div>
                {p.description && (
                  <div className="text-[10px] text-muted-foreground mt-0.5 line-clamp-2">{p.description}</div>
                )}
                {isActive && (
                  <div className="text-[9px] text-primary mt-1 font-medium">Active</div>
                )}
              </button>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
