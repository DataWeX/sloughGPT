'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Badge } from '@sloughgpt/strui'
import { useRunCompare, compareMetric } from '@/hooks/useRunCompare'
import { GitCompare, ArrowRight, Trophy } from 'lucide-react'

const METRICS: Array<{ field: 'final_loss' | 'perplexity' | 'quality_score'; label: string; format: (v: number) => string }> = [
  { field: 'final_loss', label: 'Final Loss', format: v => v?.toFixed(4) },
  { field: 'perplexity', label: 'Perplexity', format: v => v?.toFixed(2) },
  { field: 'quality_score', label: 'Quality', format: v => v?.toFixed(3) },
]

/** Side-by-side run comparison. Defaults pre-select via initialA/initialB (used by RunsSection). */
export function CompareSection({ initialA = '', initialB = '' }: { initialA?: string; initialB?: string }) {
  const cmp = useRunCompare()

  const runA = initialA || cmp.runA
  const runB = initialB || cmp.runB
  const setRunA = initialA ? undefined : cmp.setRunA
  const setRunB = initialB ? undefined : cmp.setRunB
  const result = cmp.result

  const handleCompare = () => { void cmp.comparePair(runA, runB) }

  return (
    <div className="space-y-4">
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <label className="text-sm font-medium mb-1 block">Run A</label>
              <select className="w-full border rounded-md px-3 py-2 text-sm bg-background" value={runA} onChange={e => setRunA?.(e.target.value)} disabled={cmp.fetchingOptions || !!initialA} aria-label="Run A">
                <option value="">Select run...</option>
                {cmp.runOptions.map(r => (
                  <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.model} (q={r.quality_score?.toFixed(3)})</option>
                ))}
              </select>
            </div>
            <GitCompare className="h-5 w-5 text-muted-foreground mb-2" />
            <div className="flex-1">
              <label className="text-sm font-medium mb-1 block">Run B</label>
              <select className="w-full border rounded-md px-3 py-2 text-sm bg-background" value={runB} onChange={e => setRunB?.(e.target.value)} disabled={cmp.fetchingOptions || !!initialB} aria-label="Run B">
                <option value="">Select run...</option>
                {cmp.runOptions.map(r => (
                  <option key={r.run_id} value={r.run_id}>{r.run_id} — {r.model} (q={r.quality_score?.toFixed(3)})</option>
                ))}
              </select>
            </div>
            <Button onClick={handleCompare} disabled={!runA || !runB || cmp.comparing}>
              {cmp.comparing ? 'Comparing...' : 'Compare'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <div className="space-y-4">
          <Card>
            <CardContent className="pt-6 text-center">
              <Trophy className="h-8 w-8 mx-auto mb-2 text-yellow-500" />
              <p className="text-lg font-semibold">
                {result.a_wins > result.b_wins
                  ? `Run A wins (${result.a_wins} vs ${result.b_wins})`
                  : result.b_wins > result.a_wins
                    ? `Run B wins (${result.b_wins} vs ${result.a_wins})`
                    : 'Tie'}
              </p>
            </CardContent>
          </Card>

          <div className="grid grid-cols-2 gap-4">
            {[result.run_a, result.run_b].map((run, idx) => (
              <Card key={idx}>
                <CardHeader>
                  <CardTitle className="text-sm">Run {idx === 0 ? 'A' : 'B'}: {run.run_id}</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Model</span><span>{run.model}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Dataset</span><span>{run.dataset}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Method</span><span>{run.method}</span></div>
                  {METRICS.map(m => {
                    const winner = compareMetric(m.field, result.run_a[m.field], result.run_b[m.field])
                    const mine = idx === 0 ? 'a' : 'b'
                    return (
                      <div key={m.field} className="flex justify-between">
                        <span className="text-muted-foreground">{m.label}</span>
                        <span className={winner === mine ? 'text-green-600 font-semibold' : ''}>{m.format(run[m.field])}</span>
                      </div>
                    )
                  })}
                  <div className="flex justify-between"><span className="text-muted-foreground">Time</span><span>{run.training_time_s?.toFixed(0)}s</span></div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Converged</span>
                    <Badge variant={run.converged ? 'default' : 'secondary'}>{run.converged ? 'Yes' : 'No'}</Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {Object.keys(result.differences).length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-sm">Differences</CardTitle></CardHeader>
              <CardContent>
                <div className="space-y-1">
                  {Object.entries(result.differences).map(([field, vals]) => (
                    <div key={field} className="flex items-center justify-between text-sm py-1 border-b last:border-0">
                      <span className="text-muted-foreground">{field}</span>
                      <div className="flex items-center gap-2">
                        <span className="font-mono">{String(vals.run_a)}</span>
                        <ArrowRight className="h-3 w-3 text-muted-foreground" />
                        <span className="font-mono">{String(vals.run_b)}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {!result && !cmp.comparing && (
        <Card>
          <CardContent className="py-16 text-center text-muted-foreground">
            <GitCompare className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Select two training runs to compare them side by side.</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
