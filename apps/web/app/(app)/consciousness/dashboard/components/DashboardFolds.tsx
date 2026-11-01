'use client'

// Collapsible secondary sections: consciousness-level picker and the current
// beliefs/qualia state bars. Extracted from page.tsx by the page-decomposition
// refactor — pure presentation over hook outputs.

import { Badge, Button, FoldSection } from '@sloughgpt/strui'
import { BELIEF_COLORS, QUALIA_COLORS, levelName } from '../dashboard-data'

interface ConsciousnessLevelFoldProps {
  level: number
  onLevelChange: (level: number) => void
}

export function ConsciousnessLevelFold({ level, onLevelChange }: ConsciousnessLevelFoldProps) {
  return (
    <FoldSection heading="Consciousness Level" open>
      <p className="mb-3 text-sm text-muted-foreground">
        Adjust how deeply the system reflects on each interaction
      </p>
      <div className="flex flex-wrap items-center gap-2 sm:gap-3">
        {[0, 1, 2, 3].map((lvl) => (
          <Button
            key={lvl}
            size="sm"
            variant={level === lvl ? 'default' : 'outline'}
            onClick={() => onLevelChange(lvl)}
            className="h-8 sm:h-9 w-16 sm:w-20 text-xs sm:text-sm"
          >
            {lvl === 0 ? 'Off' : lvl === 1 ? 'Basic' : lvl === 2 ? 'Full' : 'Deep'}
          </Button>
        ))}
        <Badge variant="outline" className="ml-2">
          Current: {levelName(level)}
        </Badge>
      </div>
    </FoldSection>
  )
}

interface CurrentStateFoldProps {
  currentBeliefs: Record<string, number>
  currentQualia: Record<string, number>
}

export function CurrentStateFold({ currentBeliefs, currentQualia }: CurrentStateFoldProps) {
  // Only finite numbers render bars — a shape drift upstream (e.g. an
  // envelope stored whole) skips the bad entries instead of crashing the
  // page (v.toFixed on a non-number blanked the dashboard once).
  const beliefBars = Object.entries(currentBeliefs).filter(([, v]) => Number.isFinite(v))
  const qualiaBars = Object.entries(currentQualia).filter(([, v]) => Number.isFinite(v))
  if (beliefBars.length === 0 && qualiaBars.length === 0) {
    return null
  }
  return (
    <FoldSection heading="Current State">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-6">
        {beliefBars.length > 0 && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">Beliefs</div>
            <div className="space-y-1">
              {beliefBars.map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 text-xs sm:text-sm">
                  <span className="capitalize w-20 sm:w-24">{k}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.max(0, Math.min(100, (v as number) * 100))}%`,
                        backgroundColor: BELIEF_COLORS[k] || 'rgb(var(--chart-1))',
                      }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-10 text-right">
                    {((v as number) * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        {qualiaBars.length > 0 && (
          <div>
            <div className="text-xs font-medium text-muted-foreground mb-2">Qualia</div>
            <div className="space-y-1">
              {qualiaBars.map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 text-xs sm:text-sm">
                  <span className="capitalize w-20 sm:w-24">{k}</span>
                  <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.max(0, Math.min(100, (((v as number) + 1) / 2) * 100))}%`,
                        backgroundColor: QUALIA_COLORS[k] || 'rgb(var(--chart-1))',
                      }}
                    />
                  </div>
                  <span className="text-xs text-muted-foreground w-10 text-right">
                    {(v as number).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </FoldSection>
  )
}
