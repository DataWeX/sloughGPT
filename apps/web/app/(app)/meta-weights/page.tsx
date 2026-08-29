'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { metaWeightsController, type MetaWeights, type MetaWeightStats } from '@/lib/meta-weights-controller'
import { useToastStore } from '@/lib/toast-store'

export default function MetaWeightsPage() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<MetaWeightStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [testMessage, setTestMessage] = useState('')
  const [weights, setWeights] = useState<MetaWeights | null>(null)
  const [testing, setTesting] = useState(false)

  const fetchStats = useCallback(async () => {
    try {
      setLoading(true)
      const s = await metaWeightsController.getStats()
      setStats(s)
    } catch {
      addToast('Could not load meta-weight data', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => { fetchStats() }, [fetchStats])

  const handleTest = async () => {
    if (!testMessage.trim()) return
    setTesting(true)
    try {
      const w = await metaWeightsController.getWeights(testMessage)
      setWeights(w)
    } catch {
      addToast('Could not compute meta-weights', 'error')
    } finally {
      setTesting(false)
    }
  }

  const weightBars = weights ? [
    { label: 'Temperature', value: weights.temperature, max: 2.0 },
    { label: 'Top P', value: weights.top_p, max: 1.0 },
    { label: 'Repetition Penalty', value: weights.repetition_penalty, max: 2.0 },
    { label: 'Style Bias', value: weights.style_bias, max: 1.0 },
    { label: 'Confidence Boost', value: weights.confidence_boost, max: 1.0 },
  ] : []

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); void fetchStats() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [fetchStats])

  return (
    <PageContainer
      title="Meta-Weights"
      subtitle="Feedback-driven inference tuning"
      headerRight={
        <Button size="sm" variant="outline" onClick={fetchStats} disabled={loading}>
          <IconRefresh className="h-3.5 w-3.5 mr-1" />
          Refresh
        </Button>
      }
    >
      <KpiGrid>
        <StatCard label="Status" value={loading ? '...' : (stats ? 'Active' : 'Unavailable')} />
        <StatCard label="Samples" value={loading ? '...' : String(stats?.history_length ?? 0)} />
        <StatCard label="Avg Temperature" value={stats?.avg_temperature?.toFixed(2) ?? '—'} />
        <StatCard label="Avg Top P" value={stats?.avg_top_p?.toFixed(2) ?? '—'} />
      </KpiGrid>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Test Weights</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">Enter a message to see what meta-weight adjustments the system would apply based on similar past feedback.</p>
          <div className="flex gap-2">
            <Input
              value={testMessage}
              onChange={e => setTestMessage(e.target.value)}
              placeholder="Type a message to test..."
              onKeyDown={e => e.key === 'Enter' && handleTest()}
            />
            <Button size="sm" onClick={handleTest} disabled={testing || !testMessage.trim()}>
              {testing ? 'Computing...' : 'Compute'}
            </Button>
          </div>

          {weights && (
            <div className="space-y-3 pt-2">
              <p className="text-xs text-muted-foreground">Based on {weights.based_on_samples} feedback samples</p>

              <div className="space-y-2">
                {weightBars.map(w => (
                  <div key={w.label} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium">{w.label}</span>
                      <span className="text-xs font-mono tabular-nums text-muted-foreground">{w.value.toFixed(3)}</span>
                    </div>
                    <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full transition-all"
                        style={{ width: `${Math.min(100, (w.value / w.max) * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="border rounded-md p-3">
                  <span className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Top K</span>
                  <p className="text-base font-semibold tabular-nums">{weights.top_k}</p>
                </div>
                <div className="border rounded-md p-3">
                  <span className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Based on Samples</span>
                  <p className="text-base font-semibold tabular-nums">{weights.based_on_samples}</p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">How It Works</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground space-y-2">
            <p>Meta-weights adjust inference parameters based on feedback from similar conversations. When you rate a response, the system learns which settings produce better outputs.</p>
            <ul className="list-disc list-inside space-y-1 text-xs">
              <li><strong>Temperature</strong> — controls randomness (higher = more creative)</li>
              <li><strong>Top P</strong> — nucleus sampling threshold</li>
              <li><strong>Repetition Penalty</strong> — discourages repeated phrases</li>
              <li><strong>Style Bias</strong> — shifts between formal and casual tone</li>
              <li><strong>Confidence Boost</strong> — increases certainty in responses</li>
            </ul>
          </div>
        </CardContent>
      </Card>
    </PageContainer>
  )
}
