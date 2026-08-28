'use client'

import { useState } from 'react'
import { cn } from '@sloughgpt/strui'
import TraitRadarChart from './TraitRadarChart'
import PersonalitySummary from './PersonalitySummary'

interface SoulVisualizerProps {
  traitWeights: Record<string, Record<string, number>>
  currentSoulName: string | null
}

const GROUP_COLORS: Record<string, string> = {
  personality: 'rgb(var(--primary))',
  cognition: 'rgb(var(--chart-4))',
  emotion: 'rgb(var(--destructive))',
}

const GROUP_LABELS: Record<string, string> = {
  personality: 'Personality',
  cognition: 'Cognition',
  emotion: 'Emotion',
}

export default function SoulVisualizer({ traitWeights, currentSoulName }: SoulVisualizerProps) {
  const [view, setView] = useState<'summary' | 'chart'>('summary')

  const groups = ['personality', 'cognition', 'emotion'] as const

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
