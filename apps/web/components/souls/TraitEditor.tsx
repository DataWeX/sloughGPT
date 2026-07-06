'use client'

import { useState, useMemo, useCallback } from 'react'
import { Slider } from '@sloughgpt/strui'
import { deriveArchetype } from './PersonalitySummary'

interface TraitEditorProps {
  traitWeights: Record<string, Record<string, number>>
  onSave: (weights: Record<string, Record<string, number>>) => void
  onReset: () => void
}

const GROUPS: { key: string; label: string; color: string }[] = [
  { key: 'personality', label: 'Personality', color: '#8b5cf6' },
  { key: 'cognition', label: 'Cognition', color: '#3b82f6' },
  { key: 'emotion', label: 'Emotion', color: '#ec4899' },
]

function formatTraitName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export default function TraitEditor({ traitWeights, onSave, onReset }: TraitEditorProps) {
  const [weights, setWeights] = useState(() => JSON.parse(JSON.stringify(traitWeights)) as Record<string, Record<string, number>>)
  const [dirty, setDirty] = useState(false)

  const archetype = useMemo(() => deriveArchetype(weights), [weights])

  const handleChange = useCallback((group: string, trait: string, value: number) => {
    setWeights(prev => {
      const next = { ...prev }
      next[group] = { ...next[group], [trait]: value / 100 }
      return next
    })
    setDirty(true)
  }, [])

  const handleSave = () => {
    onSave(weights)
    setDirty(false)
  }

  return (
    <div className="space-y-4" data-testid="trait-editor">
      {/* Live archetype badge */}
      <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/40 border border-border/50">
        <span className="text-[11px] text-muted-foreground">Archetype</span>
        <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary transition-all">
          {archetype.label}
        </span>
      </div>

      {/* Trait sliders by group */}
      <div className="space-y-4">
        {GROUPS.map(group => {
          const traits = weights[group.key]
          if (!traits || Object.keys(traits).length === 0) return null
          return (
            <div key={group.key}>
              <div className="flex items-center gap-1.5 mb-2">
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: group.color }} />
                <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                  {group.label}
                </span>
              </div>
              <div className="space-y-1.5">
                {Object.entries(traits).map(([name, val]) => {
                  const pct = Math.round((val ?? 0.5) * 100)
                  return (
                    <div key={name} className="flex items-center gap-2">
                      <span className="text-[10px] text-muted-foreground w-20 truncate shrink-0">
                        {formatTraitName(name)}
                      </span>
                      <Slider
                        value={[pct]}
                        onValueChange={([v]) => handleChange(group.key, name, v)}
                        min={0}
                        max={100}
                        step={1}
                        className="flex-1"
                        aria-label={`${formatTraitName(name)}: ${pct}`}
                      />
                      <span className="text-[10px] font-mono tabular-nums w-6 text-right" style={{ color: group.color }}>
                        {pct}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 pt-2 border-t border-border/40">
        <button
          type="button"
          onClick={handleSave}
          disabled={!dirty}
          className="text-[11px] px-3 py-1.5 rounded-md bg-primary text-primary-foreground font-medium disabled:opacity-40 disabled:cursor-not-allowed hover:bg-primary/90 transition-colors"
        >
          Save
        </button>
        <button
          type="button"
          onClick={() => { setWeights(JSON.parse(JSON.stringify(traitWeights))); setDirty(false); onReset() }}
          disabled={!dirty}
          className="text-[11px] px-3 py-1.5 rounded-md text-muted-foreground hover:text-foreground border border-border/60 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          Reset
        </button>
        <span className="text-[9px] text-muted-foreground/60 ml-auto">
          {archetype.description}
        </span>
      </div>
    </div>
  )
}
