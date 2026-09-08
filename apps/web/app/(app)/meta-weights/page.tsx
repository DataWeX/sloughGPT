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
        <Button size="sm" variant="outline" className="h-7 text-[11px]" onClick={fetchStats} disabled={loading}>
          <IconRefresh className="h-3 w-3 mr-1" />
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
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">Test Weights</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 px-2.5 pb-2.5">
          <p className="text-[10px] text-muted-foreground/60">Enter a message to see what meta-weight adjustments the system would apply based on similar past feedback.</p>
          <div className="flex gap-1.5">
            <Input
              value={testMessage}
              onChange={e => setTestMessage(e.target.value)}
              placeholder="Type a message to test..."
              className="h-7 text-[11px] flex-1"
              onKeyDown={e => e.key === 'Enter' && handleTest()}
            />
            <Button size="sm" className="h-7 text-[11px]" onClick={handleTest} disabled={testing || !testMessage.trim()}>
              {testing ? 'Computing...' : 'Compute'}
            </Button>
          </div>

          {weights && (
            <div className="space-y-2 pt-1">
              <p className="text-[10px] text-muted-foreground/60">Based on {weights.based_on_samples} feedback samples</p>

              <div className="space-y-1.5">
                {weightBars.map(w => (
                  <div key={w.label} className="space-y-0.5">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] font-medium">{w.label}</span>
                      <span className="text-[10px] font-mono tabular-nums text-muted-foreground/60">{w.value.toFixed(3)}</span>
                    </div>
                    <div className="h-1 bg-muted/40 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary/60 rounded-full transition-all"
                        style={{ width: `${Math.min(100, (w.value / w.max) * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-2 gap-1.5 pt-1">
                <div className="border border-border/40 rounded-lg p-2">
                  <span className="text-[9px] text-muted-foreground/60 uppercase tracking-wider font-medium">Top K</span>
                  <p className="text-[11px] font-semibold tabular-nums mt-0.5">{weights.top_k}</p>
                </div>
                <div className="border border-border/40 rounded-lg p-2">
                  <span className="text-[9px] text-muted-foreground/60 uppercase tracking-wider font-medium">Based on Samples</span>
                  <p className="text-[11px] font-semibold tabular-nums mt-0.5">{weights.based_on_samples}</p>
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2 pt-2.5 px-2.5">
          <CardTitle className="text-[11px] font-medium">How It Works</CardTitle>
        </CardHeader>
        <CardContent className="px-2.5 pb-2.5">
          <div className="text-[10px] text-muted-foreground/60 space-y-1">
            <p>Meta-weights adjust inference parameters based on feedback from similar conversations. When you rate a response, the system learns which settings produce better outputs.</p>
            <ul className="list-disc list-inside space-y-0.5 text-[10px]">
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
