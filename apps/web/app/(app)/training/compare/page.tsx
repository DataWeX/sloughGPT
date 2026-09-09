'use client'

export const dynamic = 'force-dynamic'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle, Button, Input, Badge } from '@sloughgpt/strui'
import { settingsController } from '@/lib/settings-controller'
import { GitCompare, ArrowRight, Trophy, TrendingDown, TrendingUp } from 'lucide-react'

interface RunData {
  run_id: string
  model: string
  dataset: string
  method: string
  final_loss: number
  perplexity: number
  quality_score: number
  training_time_s: number
  converged: boolean
}

interface CompareResult {
  run_a: RunData
  run_b: RunData
  differences: Record<string, { run_a: unknown; run_b: unknown }>
  a_wins: number
  b_wins: number
}

interface RunListItem {
  run_id: string
  model: string
  quality_score: number
}

export default function TrainingComparePage() {
  const [runA, setRunA] = useState('')
  const [runB, setRunB] = useState('')
  const [result, setResult] = useState<CompareResult | null>(null)
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [loading, setLoading] = useState(false)
  const [fetchingRuns, setFetchingRuns] = useState(true)

  const fetchRuns = useCallback(async () => {
    setFetchingRuns(true)
    try {
      const resp = await settingsController.exportTrainingHistory('json', 50)
      if (resp.outcomes) {
        setRuns(resp.outcomes.map((o: Record<string, unknown>) => ({
          run_id: o.run_id as string,
          model: o.model as string,
          quality_score: o.quality_score as number,
        })))
      }
    } catch (err) {
      console.error('Failed to fetch runs:', err)
    } finally {
      setFetchingRuns(false)
    }
  }, [])

  useEffect(() => { fetchRuns() }, [fetchRuns])

  const handleCompare = async () => {
    if (!runA || !runB) return
    setLoading(true)
    try {
      const resp = await settingsController.compareTrainingRuns(runA, runB)
      if (resp.run_a) {
        setResult(resp as unknown as CompareResult)
      }
    } catch (err) {
      console.error('Failed to compare:', err)
    } finally {
      setLoading(false)
    }
  }

  const isBetter = (field: string, valA: number, valB: number) => {
    const lowerIsBetter = ['final_loss', 'perplexity', 'training_time_s']
    if (lowerIsBetter.includes(field)) {
      return valA < valB ? 'a' : valB < valA ? 'b' : 'tie'
    }
    return valA > valB ? 'a' : valB > valA ? 'b' : 'tie'
  }

  return (
    <PageContainer title="Training Comparison">
      <AppRouteHeader left={<AppRouteHeaderLead title="Training Comparison" />} />

      {/* Selector */}
      <Card className="mb-6">
        <CardContent className="pt-6">
          <div className="flex items-end gap-4">
            <div className="flex-1">
              <label className="text-sm font-medium mb-1 block">Run A</label>
              <select
                className="w-full border rounded-md px-3 py-2 text-sm bg-background"
                value={runA}
                onChange={e => setRunA(e.target.value)}
                disabled={fetchingRuns}
              >
                <option value="">Select run...</option>
                {runs.map(r => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} — {r.model} (q={r.quality_score?.toFixed(3)})
                  </option>
                ))}
              </select>
            </div>
            <GitCompare className="h-5 w-5 text-muted-foreground mb-2" />
            <div className="flex-1">
              <label className="text-sm font-medium mb-1 block">Run B</label>
              <select
                className="w-full border rounded-md px-3 py-2 text-sm bg-background"
                value={runB}
                onChange={e => setRunB(e.target.value)}
                disabled={fetchingRuns}
              >
                <option value="">Select run...</option>
                {runs.map(r => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.run_id} — {r.model} (q={r.quality_score?.toFixed(3)})
                  </option>
                ))}
              </select>
            </div>
            <Button onClick={handleCompare} disabled={!runA || !runB || loading}>
              {loading ? 'Comparing...' : 'Compare'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Winner banner */}
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

          {/* Side by side */}
          <div className="grid grid-cols-2 gap-4">
            {[result.run_a, result.run_b].map((run, idx) => (
              <Card key={idx}>
                <CardHeader>
                  <CardTitle className="text-sm">
                    Run {idx === 0 ? 'A' : 'B'}: {run.run_id}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between"><span className="text-muted-foreground">Model</span><span>{run.model}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Dataset</span><span>{run.dataset}</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Method</span><span>{run.method}</span></div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Final Loss</span>
                    <span className={isBetter('final_loss', result.run_a.final_loss, result.run_b.final_loss) === (idx === 0 ? 'a' : 'b') ? 'text-green-600 font-semibold' : ''}>
                      {run.final_loss?.toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Perplexity</span>
                    <span className={isBetter('perplexity', result.run_a.perplexity, result.run_b.perplexity) === (idx === 0 ? 'a' : 'b') ? 'text-green-600 font-semibold' : ''}>
                      {run.perplexity?.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Quality</span>
                    <span className={isBetter('quality_score', result.run_a.quality_score, result.run_b.quality_score) === (idx === 0 ? 'a' : 'b') ? 'text-green-600 font-semibold' : ''}>
                      {run.quality_score?.toFixed(3)}
                    </span>
                  </div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Time</span><span>{run.training_time_s?.toFixed(0)}s</span></div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Converged</span>
                    <Badge variant={run.converged ? 'default' : 'secondary'}>{run.converged ? 'Yes' : 'No'}</Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Differences */}
          {Object.keys(result.differences).length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Differences</CardTitle>
              </CardHeader>
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

      {!result && !loading && (
        <Card>
          <CardContent className="py-16 text-center text-muted-foreground">
            <GitCompare className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Select two training runs to compare them side by side.</p>
          </CardContent>
        </Card>
      )}
    </PageContainer>
  )
}
