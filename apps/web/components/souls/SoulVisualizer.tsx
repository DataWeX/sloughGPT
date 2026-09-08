'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import TraitRadarChart from './TraitRadarChart'
import PersonalitySummary from './PersonalitySummary'
import { SOUL_GROUP_COLORS, SOUL_GROUP_LABELS, SOUL_GROUP_KEYS } from './soul-constants'

interface SoulVisualizerProps {
  traitWeights: Record<string, Record<string, number>>
  currentSoulName: string | null
}

const GROUP_COLORS = SOUL_GROUP_COLORS
const GROUP_LABELS = SOUL_GROUP_LABELS

export default function SoulVisualizer({ traitWeights, currentSoulName }: SoulVisualizerProps) {
  const [view, setView] = useState<'summary' | 'chart'>('summary')

  const groups = SOUL_GROUP_KEYS

  if (!traitWeights || Object.keys(traitWeights).length === 0) return null

  const hasRadarData = groups.some(g => {
    const t = traitWeights[g]
    return t && typeof t === 'object' && Object.keys(t).length > 0
  })

  return (
    <>
      {/* ── View toggle ── */}
      {hasRadarData && (
        <div className="flex flex-wrap items-center gap-1 mb-3">
          <button
            type="button"
            className={cn('text-[10px] px-2 py-1 rounded-md transition-colors', view === 'summary' ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:text-foreground')}
            onClick={() => setView('summary')}
          >
            List
          </button>
          <button
            type="button"
            className={cn('text-[10px] px-2 py-1 rounded-md transition-colors', view === 'chart' ? 'bg-primary/10 text-primary font-medium' : 'text-muted-foreground hover:text-foreground')}
            onClick={() => setView('chart')}
          >
            Radar
          </button>
        </div>
      )}

      {/* ── Summary view (default) ── */}
      {view === 'summary' && (
        <div key="summary" className="view-pane">
          <PersonalitySummary traitWeights={traitWeights} currentSoulName={currentSoulName} />
        </div>
      )}

      {/* ── Radar chart view ── */}
      {view === 'chart' && hasRadarData && (
        <div key="chart" className="grid grid-cols-1 sm:grid-cols-3 gap-4 view-pane">
          {groups.map((group, idx) => {
            const traits = traitWeights[group] as Record<string, number> | undefined
            if (!traits || typeof traits !== 'object') return null
            const entries = Object.entries(traits)
            if (entries.length === 0) return null
            return (
              <div key={group} style={{ animationDelay: `${idx * 100}ms` }}>
                <TraitRadarChart
                  data={traits}
                  label={GROUP_LABELS[group]}
                  color={GROUP_COLORS[group]}
                />
              </div>
            )
          })}
        </div>
      )}
    </>
  )
}
