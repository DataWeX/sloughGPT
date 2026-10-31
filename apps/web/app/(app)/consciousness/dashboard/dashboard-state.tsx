'use client'

/**
 * The two collapsible secondary sections: consciousness level controls and the
 * Current State bars (beliefs + qualia).
 *
 * The bar widths are derived here from the raw current-state maps, including
 * the qualia [-1, 1] → [0, 100]% mapping, so the page only forwards state.
 */
import { Badge, Button, FoldSection } from '@sloughgpt/strui'
import { BELIEF_COLORS, QUALIA_COLORS, levelName } from './dashboard-format'

interface DashboardStateProps {
  consciousnessLevel: number
  currentBeliefs: Record<string, number>
  currentQualia: Record<string, number>
  onLevelChange: (level: number) => void
}

const LEVELS = [0, 1, 2, 3]

export function DashboardState({
  consciousnessLevel,
  currentBeliefs,
  currentQualia,
  onLevelChange,
}: DashboardStateProps) {
  const hasState =
    Object.keys(currentBeliefs).length > 0 || Object.keys(currentQualia).length > 0

  return (
    <>
      {/* Secondary controls — collapsible so the body stays focused */}
      <FoldSection heading="Consciousness Level" open>
        <p className="mb-3 text-sm text-muted-foreground">
          Adjust how deeply the system reflects on each interaction
        </p>
        <div className="flex flex-wrap items-center gap-2 sm:gap-3">
          {LEVELS.map((lvl) => (
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

      {hasState && (
        <FoldSection heading="Current State">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-6">
            {Object.keys(currentBeliefs).length > 0 && (
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
                            width: `${Math.max(0, Math.min(100, v * 100))}%`,
                            backgroundColor: BELIEF_COLORS[k] || 'rgb(var(--chart-1))',
                          }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right">
                        {(v * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {Object.keys(currentQualia).length > 0 && (
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
                            width: `${Math.max(0, Math.min(100, ((v + 1) / 2) * 100))}%`,
                            backgroundColor: QUALIA_COLORS[k] || 'rgb(var(--chart-1))',
                          }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground w-10 text-right">
                        {v.toFixed(2)}
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
