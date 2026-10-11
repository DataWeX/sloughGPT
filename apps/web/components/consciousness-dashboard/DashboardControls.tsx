/**
 * Secondary controls — the collapsible consciousness-level picker and the
 * current beliefs/qualia state bars.
 *
 * Both are `FoldSection`s so the body stays focused; the state panel renders
 * nothing at all when there is no state to show.
 */

import { Badge, Button, FoldSection } from '@sloughgpt/strui'
import { BELIEF_COLORS, QUALIA_COLORS, levelName } from './types'

export interface DashboardControlsProps {
  consciousnessLevel: number
  onLevelChange: (level: number) => void
  currentBeliefs: Record<string, number>
  currentQualia: Record<string, number>
}

export function DashboardControls({
  consciousnessLevel,
  onLevelChange,
  currentBeliefs,
  currentQualia,
}: DashboardControlsProps) {
  const hasBeliefs = Object.keys(currentBeliefs).length > 0
  const hasQualia = Object.keys(currentQualia).length > 0

  return (
    <>
      <FoldSection heading="Consciousness Level" open>
        <p className="mb-3 text-sm text-muted-foreground">
          Adjust how deeply the system reflects on each interaction
        </p>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {[0, 1, 2, 3].map((lvl) => (
            <Button
              key={lvl}
              size="sm"
              variant={consciousnessLevel === lvl ? 'default' : 'outline'}
              onClick={() => onLevelChange(lvl)}
              className="h-8 sm:h-9 w-16 sm:w-20 text-xs sm:text-sm"
            >
              {lvl === 0 ? 'Off' : lvl === 1 ? 'Basic' : lvl === 2 ? 'Full' : 'Deep'}
            </Button>
          ))}
          <Badge variant="outline" className="ml-2">
            Current: {levelName(consciousnessLevel)}
          </Badge>
        </div>
      </FoldSection>

      {(hasBeliefs || hasQualia) && (
        <FoldSection heading="Current State">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-6">
            {hasBeliefs && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2">Beliefs</div>
                <div className="space-y-1">
                  {Object.entries(currentBeliefs).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-2 text-xs sm:text-sm">
                      <span className="capitalize w-20 sm:w-24">{k}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, (v as number) * 100))}%`,
                            backgroundColor: BELIEF_COLORS[k] || 'var(--chart-1)',
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
            {hasQualia && (
              <div>
                <div className="text-xs font-medium text-muted-foreground mb-2">Qualia</div>
                <div className="space-y-1">
                  {Object.entries(currentQualia).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-2 text-xs sm:text-sm">
                      <span className="capitalize w-20 sm:w-24">{k}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.max(0, Math.min(100, (((v as number) + 1) / 2) * 100))}%`,
                            backgroundColor: QUALIA_COLORS[k] || 'var(--chart-1)',
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
      )}
    </>
  )
}
