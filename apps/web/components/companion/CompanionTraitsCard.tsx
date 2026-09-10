'use client'

import { useState, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'
import type { CompanionTraits } from '@/lib/companion-controller'

const TRAIT_META: Record<string, { label: string; color: string; icon: string }> = {
  warmth: { label: 'Warmth', color: 'bg-warning', icon: '☀️' },
  curiosity: { label: 'Curiosity', color: 'bg-primary', icon: '🔍' },
  creativity: { label: 'Creativity', color: 'bg-accent', icon: '🎨' },
  confidence: { label: 'Confidence', color: 'bg-success', icon: '💪' },
  humor: { label: 'Humor', color: 'bg-destructive', icon: '😄' },
}

interface CompanionTraitsCardProps {
  traits: CompanionTraits | null
  onSave?: (traits: CompanionTraits) => void
  onReset?: () => void
}

export function CompanionTraitsCard({ traits, onSave, onReset }: CompanionTraitsCardProps) {
  const [draft, setDraft] = useState<CompanionTraits | null>(null)
  const [saving, setSaving] = useState(false)
  const current = draft ?? traits

  const handleChange = useCallback((key: string, value: number) => {
    if (!current) return
    setDraft({ ...current, [key]: value })
  }, [current])

  const handleSave = async () => {
    if (!draft) return
    setSaving(true)
    try { onSave?.(draft) } finally { setSaving(false) }
  }

  const traitEntries = current
    ? (Object.entries(current) as [string, number][]).filter(([k]) => k !== 'name')
    : []

  const avg = traitEntries.length > 0
    ? traitEntries.reduce((s, [, v]) => s + v, 0) / traitEntries.length
    : 0

  return (
    <Card data-testid="companion-traits">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Personality Traits</CardTitle>
          <div className="flex gap-1">
            {onReset && (
              <Button size="sm" variant="ghost" className="text-destructive text-[10px]" onClick={onReset}>
                Reset
              </Button>
            )}
            {draft && (
              <Button size="sm" className="text-[10px]" onClick={handleSave} disabled={saving}>
                {saving ? 'Saving...' : 'Save'}
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {!current && <div className="text-sm text-muted-foreground">No traits loaded.</div>}
        {current && (
          <>
            <div className="flex items-center gap-3 mb-4">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">Overall</div>
              <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                <div className="h-full bg-primary rounded-full transition-all" style={{ width: `${avg * 100}%` }} />
              </div>
              <div className="text-xs font-medium">{(avg * 100).toFixed(0)}%</div>
            </div>
            <div className="space-y-3">
              {traitEntries.map(([key, value]) => {
                const meta = TRAIT_META[key] ?? { label: key, color: 'bg-muted', icon: '•' }
                return (
                  <div key={key}>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1.5">
                        <span className="text-xs">{meta.icon}</span>
                        <span className="text-xs font-medium">{meta.label}</span>
                      </div>
                      <span className="text-[10px] text-muted-foreground">{(value * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={value}
                      onChange={e => handleChange(key, parseFloat(e.target.value))}
                      className="w-full h-1.5 accent-primary"
                      data-testid={`trait-${key}`}
                    />
                  </div>
                )
              })}
            </div>
          </>
        )}
      </CardContent>
    </Card>
  )
}
