'use client'

/** Analytics tab: soul overview, personality comparison radar and checkpoint summary. */
import { Card, CardContent, CardHeader, CardTitle, TabsContent, cn } from '@sloughgpt/strui'
import { formatShortDate } from '@/lib/time-format'
import { traitColor, traitLabel } from './souls-format'
import { TraitRadar } from './souls-panels'
import type { SoulsWorkbench } from './useSoulsWorkbench'

export function AnalyticsTab({ wb }: { wb: SoulsWorkbench }) {
  const { checkpoints, currentSoul, souls } = wb

  return (
    <TabsContent value="analytics" className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Soul Overview</CardTitle>
        </CardHeader>
        <CardContent>
          {souls.length === 0 ? (
            <div className="text-center py-8 space-y-2">
              <p className="text-sm text-muted-foreground">No souls available for analysis.</p>
              <p className="text-xs text-muted-foreground">
                Register a soul from a <code className="bg-muted px-1 rounded">.soul</code> file to
                see analytics here.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Per-soul summary */}
              <div className="space-y-2">
                {souls.map((soul) => {
                  const avgPersonality =
                    soul.personality && Object.keys(soul.personality).length > 0
                      ? Object.values(soul.personality).reduce((a, b) => a + b, 0) /
                        Object.values(soul.personality).length
                      : 0
                  return (
                    <div
                      key={soul.name}
                      className="flex items-center gap-3 rounded-md border border-border/40 px-3 py-2 text-sm"
                    >
                      <TraitRadar values={soul.personality || {}} size={40} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{soul.name}</span>
                          {currentSoul === soul.name && (
                            <span className="text-xs px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
                              active
                            </span>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {soul.lineage || 'Unknown lineage'} ·{' '}
                          {soul.size_mb != null ? `${soul.size_mb.toFixed(1)} MB` : '—'}
                          {soul.born_at ? ` · ${formatShortDate(soul.born_at)}` : ''}
                        </div>
                      </div>
                      <div className="text-right">
                        <div
                          className={cn(
                            'text-sm font-mono font-medium',
                            traitColor(avgPersonality),
                          )}
                        >
                          {(avgPersonality * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-muted-foreground">avg personality</div>
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Personality comparison chart */}
              {souls.length > 1 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                    Personality Comparison
                  </div>
                  <div className="grid grid-cols-1 gap-1">
                    {(() => {
                      const allTraits = new Set<string>()
                      souls.forEach(
                        (s) =>
                          s.personality &&
                          Object.keys(s.personality).forEach((t) => allTraits.add(t)),
                      )
                      const traits = Array.from(allTraits).slice(0, 10)
                      return traits.map((trait) => (
                        <div key={trait} className="flex items-center gap-2 text-xs">
                          <span className="w-24 text-muted-foreground truncate">
                            {traitLabel(trait)}
                          </span>
                          <div className="flex-1 flex gap-1">
                            {souls.map((soul) => {
                              const val = soul.personality?.[trait] ?? 0
                              return (
                                <div key={soul.name} className="flex-1">
                                  <div className="h-2 rounded-full bg-muted/50 overflow-hidden">
                                    <div
                                      className="h-full rounded-full bg-primary/60 transition-all"
                                      style={{
                                        width: `${Math.max(0, Math.min(100, val * 100))}%`,
                                      }}
                                    />
                                  </div>
                                </div>
                              )
                            })}
                          </div>
                        </div>
                      ))
                    })()}
                  </div>
                  <div className="flex gap-3 mt-2 text-xs text-muted-foreground">
                    {souls.map((soul) => (
                      <span key={soul.name} className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full bg-primary/60" />
                        {soul.name}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Checkpoint summary */}
              {checkpoints.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wider">
                    Checkpoint Summary
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    <div className="rounded-md border border-border/40 p-2 text-center">
                      <div className="text-base font-bold">{checkpoints.length}</div>
                      <div className="text-xs text-muted-foreground">Total</div>
                    </div>
                    <div className="rounded-md border border-border/40 p-2 text-center">
                      <div className="text-base font-bold text-success">
                        {checkpoints.filter((c) => c.verdict === 'improved').length}
                      </div>
                      <div className="text-xs text-muted-foreground">Improved</div>
                    </div>
                    <div className="rounded-md border border-border/40 p-2 text-center">
                      <div className="text-base font-bold text-destructive">
                        {checkpoints.filter((c) => c.verdict === 'degraded').length}
                      </div>
                      <div className="text-xs text-muted-foreground">Degraded</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </TabsContent>
  )
}
