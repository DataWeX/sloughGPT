'use client'

import { Card, CardHeader, CardTitle, CardContent, cn } from '@sloughgpt/strui'
import type { ProbeStanding, TrainedOnEntry } from '@/lib/training-controller'

interface ProbeStandingsCardProps {
  standings: ProbeStanding[]
  trainedOn?: TrainedOnEntry[]
}

function firstSeen(standings: ProbeStanding[], ref: string): number | undefined {
  for (const s of standings) {
    const v = s.probes[ref]
    if (typeof v === 'number') return v
  }
  return undefined
}

export function ProbeStandingsCard({ standings, trainedOn }: ProbeStandingsCardProps) {
  if (!standings.length) return null

  const refs: string[] = []
  for (const s of standings) {
    for (const r of Object.keys(s.probes)) {
      if (!refs.includes(r)) refs.push(r)
    }
  }

  const drift: Record<string, number> = {}
  for (const ref of refs) {
    const seen = standings
      .map((s) => s.probes[ref])
      .filter((v): v is number => typeof v === 'number')
    if (seen.length >= 2) drift[ref] = seen[seen.length - 1] - seen[0]
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">Probe standings</CardTitle>
          <span className="text-[10px] text-muted-foreground">
            held-out loss per stage · ↑ = forgot
          </span>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="py-1 pr-3 font-medium">Stage</th>
                <th className="py-1 pr-3 text-right font-medium">Step</th>
                {refs.map((r) => (
                  <th key={r} className="px-3 py-1 text-right font-medium">
                    {r}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {standings.map((s) => (
                <tr key={s.stage} className="border-t border-border/40">
                  <td className="py-1.5 pr-3 font-mono">
                    {s.stage + 1}: {s.dataset}
                  </td>
                  <td className="py-1.5 pr-3 text-right tabular-nums text-muted-foreground">
                    {s.global_step}
                  </td>
                  {refs.map((r) => {
                    const v = s.probes[r]
                    const base = firstSeen(standings, r)
                    const delta =
                      typeof v === 'number' && typeof base === 'number' ? v - base : null
                    const notYet = typeof v !== 'number'
                    return (
                      <td
                        key={r}
                        className={cn(
                          'px-3 py-1.5 text-right font-mono tabular-nums',
                          notYet && 'text-muted-foreground/40',
                        )}
                      >
                        {notYet ? (
                          '—'
                        ) : (
                          <>
                            {v.toFixed(3)}
                            {delta !== null && Math.abs(delta) > 1e-6 && (
                              <span
                                className={cn(
                                  'ml-1.5 text-[10px]',
                                  delta > 0 ? 'text-destructive' : 'text-success',
                                )}
                              >
                                {delta > 0 ? '↑' : '↓'}
                                {Math.abs(delta) > 0.0005 ? Math.abs(delta).toFixed(3) : ''}
                              </span>
                            )}
                          </>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {refs.some((r) => drift[r] !== undefined) && (
          <div className="flex flex-wrap gap-2">
            {refs
              .filter((r) => drift[r] !== undefined)
              .map((r) => {
                const d = drift[r]
                const forgot = d > 1e-6
                return (
                  <span
                    key={r}
                    className={cn(
                      'px-2 py-0.5 rounded text-[10px] font-medium',
                      forgot ? 'bg-destructive/10 text-destructive' : 'bg-success/10 text-success',
                    )}
                  >
                    {r} {forgot ? '↑ forgot +' : '↓ held'}
                    {forgot ? d.toFixed(3) : ` -${Math.abs(d).toFixed(3)}`}
                  </span>
                )
              })}
          </div>
        )}

        {trainedOn && trainedOn.length > 0 && (
          <p className="text-[10px] font-mono text-muted-foreground">
            trained on: {trainedOn.map((t) => t.ref).join(' → ')}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
