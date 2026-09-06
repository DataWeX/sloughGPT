'use client'

import { Card, CardContent, CardHeader, CardTitle, Button, Input } from '@sloughgpt/strui'
import { IconRefresh, IconDownload } from '@sloughgpt/strui'
import ModelsCard from '@/components/compare/ModelsCard'
import ComparisonTableCard from '@/components/compare/ComparisonTableCard'
import SummaryCard from '@/components/compare/SummaryCard'
import { ModelComparisonInsightsCard } from '@/components/compare/ModelComparisonInsightsCard'
import dynamicNext from 'next/dynamic'
import type { ModelEntry, SavedSnapshot } from '@/hooks/useComparison'
import type { BenchmarkResult } from '@/lib/benchmark-controller'

const OutputComparisonCard = dynamicNext<{ models: ModelEntry[] }>(() => import('@/components/compare/OutputComparisonCard') as Promise<{ default: React.ComponentType<{ models: ModelEntry[] }> }>, { ssr: false })
const VisualComparisonCard = dynamicNext(() => import('@/components/compare/VisualComparisonCard'), { ssr: false })

interface ComparisonViewProps {
  models: ModelEntry[]
  loading: boolean
  results: Record<string, BenchmarkResult | null>
  running: Set<string>
  snapshots: SavedSnapshot[]
  snapshotName: string
  onSnapshotNameChange: (name: string) => void
  completedResults: [string, BenchmarkResult][]
  bestMetrics: Record<string, number>
  chartData: { name: string; throughput: number; latency: number; memory: number }[]
  onBenchmark: (modelId: string) => void
  onClear: (modelId: string) => void
  onRunAll: () => void
  onExport: () => void
  onSaveSnapshot: () => void
  onLoadSnapshot: (snap: SavedSnapshot) => void
  onDeleteSnapshot: (id: string) => void
}

export function ComparisonView({
  models,
  loading,
  results,
  running,
  snapshots,
  snapshotName,
  onSnapshotNameChange,
  completedResults,
  bestMetrics,
  chartData,
  onBenchmark,
  onClear,
  onRunAll,
  onExport,
  onSaveSnapshot,
  onLoadSnapshot,
  onDeleteSnapshot,
}: ComparisonViewProps) {
  return (
    <>
      {snapshots.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Saved Comparisons</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {snapshots.map(snap => (
                <div key={snap.id} className="flex items-center gap-1 rounded-lg border border-border/40 bg-muted/20 px-2 py-1">
                  <button type="button" onClick={() => onLoadSnapshot(snap)} className="text-xs font-medium hover:text-primary transition-colors">
                    {snap.name}
                  </button>
                  <span className="text-xs text-muted-foreground">{new Date(snap.savedAt).toLocaleDateString()}</span>
                  <button type="button" onClick={() => onDeleteSnapshot(snap.id)} aria-label={`Delete snapshot ${snap.name}`} className="text-xs text-muted-foreground hover:text-destructive ml-1">×</button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <ModelsCard models={models} loading={loading} results={results} running={running} onBenchmark={onBenchmark} onClear={onClear} />

      {completedResults.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center space-y-3">
            <p className="text-sm text-muted-foreground">No benchmark results yet.</p>
            <p className="text-xs text-muted-foreground/70 max-w-md mx-auto">
              Run benchmarks on your models to see side-by-side comparisons. Click &ldquo;Benchmark all&rdquo; or use the benchmark button on each model card above.
            </p>
            <Button size="sm" variant="outline" className="h-8 text-xs" onClick={onRunAll} disabled={loading || models.length === 0}>
              Benchmark all
            </Button>
            <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground/50 pt-2">
              <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">R</kbd> Benchmark all</span>
              <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">Ctrl+S</kbd> Save snapshot</span>
              <span><kbd className="px-1 py-0.5 rounded bg-muted/50 border border-border/50 font-mono">Ctrl+E</kbd> Export</span>
            </div>
          </CardContent>
        </Card>
      ) : (
        <>
          <ComparisonTableCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
          <ModelComparisonInsightsCard completedResults={completedResults} models={models} bestMetrics={bestMetrics} />
          <SummaryCard completedResults={completedResults} models={models} />
          <OutputComparisonCard models={models} />
          <VisualComparisonCard chartData={chartData} />
        </>
      )}
    </>
  )
}

interface ComparisonHeaderProps {
  completedResults: [string, BenchmarkResult][]
  snapshotName: string
  onSnapshotNameChange: (name: string) => void
  onSaveSnapshot: () => void
  onExport: () => void
  onRunAll: () => void
  loading: boolean
  running: Set<string>
}

export function ComparisonHeader({
  completedResults,
  snapshotName,
  onSnapshotNameChange,
  onSaveSnapshot,
  onExport,
  onRunAll,
  loading,
  running,
}: ComparisonHeaderProps) {
  return (
    <div className="flex items-center gap-2">
      {completedResults.length > 0 && (
        <>
          <div className="flex items-center gap-1">
            <Input
              value={snapshotName}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => onSnapshotNameChange(e.target.value)}
              placeholder="Snapshot name..."
              aria-label="Snapshot name"
              className="h-8 w-40 text-xs"
              onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') onSaveSnapshot() }}
            />
            <Button variant="outline" size="sm" onClick={onSaveSnapshot}>Save</Button>
          </div>
          <Button variant="outline" size="sm" onClick={onExport}>
            <IconDownload className="h-3.5 w-3.5 mr-1" />
            Export
          </Button>
        </>
      )}
      <Button variant="outline" size="sm" onClick={onRunAll} disabled={loading || running.size > 0}>
        <IconRefresh className="h-3.5 w-3.5 mr-1" /> Benchmark all
      </Button>
    </div>
  )
}
