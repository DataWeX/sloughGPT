'use client'

import { useCallback, useEffect, useState } from 'react'
import { Button } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { trainingJobsController } from '@/lib/training-controller'
import { devDebug } from '@/lib/dev-log'
import { useToastStore } from '@/lib/toast-store'

export function ConversationDataSection() {
  const addToast = useToastStore(s => s.addToast)
  const [stats, setStats] = useState<{
    total_pairs: number
    positive_pairs: number
    negative_pairs: number
    neutral_pairs: number
    unused_pairs: number
  } | null>(null)
  const [loading, setLoading] = useState(true)
  const [exporting, setExporting] = useState(false)
  const [strategy, setStrategy] = useState<'balanced' | 'weighted' | 'simple'>('balanced')
  const [targetCount, setTargetCount] = useState(100)

  const fetchStats = useCallback(async () => {
    setLoading(true)
    try {
      const res = await trainingJobsController.getStatus() as { total_jobs: number; completed_jobs: number; running_jobs?: unknown[]; failed_jobs?: number }
      setStats({
        total_pairs: res.total_jobs,
        positive_pairs: res.completed_jobs,
        negative_pairs: (res.running_jobs as unknown[])?.length || 0,
        neutral_pairs: res.failed_jobs || 0,
        unused_pairs: 0,
      })
    } catch (err) {
      devDebug('Failed to fetch training stats:', err)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchStats()
  }, [fetchStats])

  const handleExport = useCallback(async () => {
    setExporting(true)
    try {
      const data = await trainingJobsController.exportFeedbackPairs(0, targetCount) as { pairs_count?: number; filepath?: string; error?: string }

      if (data.error) {
        addToast(String(data.error), 'error')
      } else {
        addToast(`Exported ${data.pairs_count || 0} pairs to ${data.filepath || 'file'}`, 'success')
        void fetchStats()
      }
    } catch (err) {
      console.error('Export error:', err)
      addToast('Something went wrong exporting', 'error')
    } finally {
      setExporting(false)
    }
  }, [targetCount, addToast, fetchStats])

  return (
    <div className="space-y-4">
      {loading ? (
        <div className="text-center py-4 text-muted-foreground">Loading...</div>
      ) : stats ? (
        <>
          <div className="grid grid-cols-2 gap-2">
            <div className="p-3 rounded-lg bg-success/10 border border-success/30">
              <div className="text-base font-semibold text-success">{stats.positive_pairs}</div>
              <div className="text-xs text-muted-foreground">Positive (👍)</div>
            </div>
            <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30">
              <div className="text-base font-semibold text-destructive">{stats.negative_pairs}</div>
              <div className="text-xs text-muted-foreground">Negative (👎)</div>
            </div>
            <div className="p-3 rounded-lg bg-muted">
              <div className="text-base font-semibold">{stats.neutral_pairs}</div>
              <div className="text-xs text-muted-foreground">Neutral</div>
            </div>
            <div className="p-3 rounded-lg bg-muted">
              <div className="text-base font-semibold">{stats.total_pairs}</div>
              <div className="text-xs text-muted-foreground">Total Pairs</div>
            </div>
          </div>

          <div className="space-y-3">
            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Sampling Strategy</label>
              <Select value={strategy} onValueChange={(v: string) => setStrategy(v as 'balanced' | 'weighted' | 'simple')}>
                <SelectTrigger className="text-sm">
                  <SelectValue placeholder="Select strategy..." />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="balanced">Balanced (equal +/ /neutral)</SelectItem>
                  <SelectItem value="weighted">Weighted (edge cases)</SelectItem>
                  <SelectItem value="simple">Simple (filter by quality)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-xs text-muted-foreground mb-1 block">Target Count</label>
              <input
                type="number"
                value={targetCount}
                onChange={(e) => setTargetCount(parseInt(e.target.value) || 100)}
                min={10}
                max={1000}
                className="sl-input py-1.5 text-sm"
              />
            </div>

            <Button
              onClick={handleExport}
              disabled={exporting || (stats?.total_pairs ?? 0) < 5}
              className="w-full"
              size="sm"
            >
              {exporting ? 'Exporting...' : `Export ${targetCount} Pairs for Training`}
            </Button>

            {(stats?.total_pairs ?? 0) < 5 && (
              <p className="text-xs text-warning">
                Need at least 5 conversation pairs to export. Chat more to build your training data!
              </p>
            )}
          </div>
        </>
      ) : (
        <div className="text-center py-4 text-muted-foreground">
          No training data yet. Start chatting to build your dataset.
        </div>
      )}
    </div>
  )
}
